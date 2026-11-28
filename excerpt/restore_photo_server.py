# AI 오래된 사진 복원 (AIIntegratedServer/server.py 중 일부)

# 복원용 예외 클래스
class ReplicateServiceError(RuntimeError):
    pass


class ReplicateRequestCancelled(RuntimeError):
    pass


# 컬러화·디테일 보정 모델 불러오기
def _get_colorization_net():
    global _colorization_net
    if _colorization_net is None:
        with _colorization_lock:
            if _colorization_net is None:
                prototxt = _download_model("colorization_deploy_v2.prototxt")
                weights = _download_model("colorization_release_v2.caffemodel")
                points_path = _download_model("pts_in_hull.npy")

                net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
                points = np.load(points_path).transpose().reshape(2, 313, 1, 1)
                net.getLayer(net.getLayerId("class8_ab")).blobs = [
                    points.astype(np.float32)
                ]
                net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
                    np.full((1, 313), 2.606, dtype=np.float32)
                ]
                _colorization_net = net
    return _colorization_net


def _get_detail_net():
    global _detail_net
    if _detail_net is None:
        with _detail_lock:
            if _detail_net is None:
                model_path = _download_model("FSRCNN_x2.pb")
                net = cv2.dnn_superres.DnnSuperResImpl_create()
                net.readModel(str(model_path))
                net.setModel("fsrcnn", 2)
                _detail_net = net
    return _detail_net



# 로컬 복원 (손상·바램 복구, 컬러화, FSRCNN 디테일 보정)
def _is_grayscale_or_faded(bgr: np.ndarray) -> bool:
    channel_range = np.max(bgr, axis=2) - np.min(bgr, axis=2)
    if float(np.mean(channel_range)) < 8.0:
        return True

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    if (
        float(np.mean(saturation)) < 18.0
        and float(np.percentile(saturation, 90)) < 35.0
    ):
        return True

    # 세피아 사진은 RGB 값의 차이는 크지만 세 채널이 거의 같은 명암 형태로
    # 움직이고 Lab 색차 분산이 작습니다. 단순 채널 차이만 보면 컬러사진으로
    # 오판하므로 두 조건을 함께 사용해 단색으로 염색된 오래된 사진도 찾습니다.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    chroma_std = np.std(lab[:, :, 1:].astype(np.float32), axis=(0, 1))
    sampled = bgr[::4, ::4].reshape(-1, 3).astype(np.float32)
    channel_std = np.std(sampled, axis=0)
    if sampled.shape[0] < 4 or float(np.min(channel_std)) < 1.0:
        return False

    correlation = np.corrcoef(sampled, rowvar=False)
    channel_pairs = correlation[np.triu_indices(3, 1)]
    minimum_correlation = float(np.nanmin(channel_pairs))
    sepia_monochrome = (
        float(np.max(chroma_std)) < 7.0
        and minimum_correlation > 0.96
        and float(np.mean(saturation)) < 115.0
    )
    return sepia_monochrome


def _protect_detected_faces(mask: np.ndarray, gray: np.ndarray) -> None:
    """자동 균열 마스크가 얼굴과 머리 장식을 바꾸지 않도록 보호합니다."""
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return

        height, width = gray.shape
        max_face = max(40, min(width, height) // 3)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(24, 24),
            maxSize=(max_face, max_face),
        )
        for x, y, face_width, face_height in faces:
            horizontal_padding = max(5, round(face_width * 0.30))
            top_padding = max(6, round(face_height * 0.55))
            bottom_padding = max(5, round(face_height * 0.35))
            left = max(0, x - horizontal_padding)
            top = max(0, y - top_padding)
            right = min(width - 1, x + face_width + horizontal_padding)
            bottom = min(height - 1, y + face_height + bottom_padding)
            cv2.rectangle(mask, (left, top), (right, bottom), 0, thickness=-1)
    except (AttributeError, cv2.error, OSError):
        # 얼굴 검출은 보호 장치일 뿐이므로 검출기 문제로 전체 복원을 중단하지 않습니다.
        return


def _select_scratch_components(
    binary: np.ndarray,
    longest_side: int,
    *,
    strict: bool,
) -> np.ndarray:
    """점 형태의 질감은 버리고 길고 가는 균열 성분만 남깁니다."""
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    selected = np.zeros_like(binary)
    minimum_diagonal = max(7.0, longest_side * (0.020 if strict else 0.015))
    maximum_area = max(32, round(binary.size * 0.0025))

    for label in range(1, component_count):
        _, _, width, height, area = stats[label]
        if area < 3 or area > maximum_area:
            continue

        diagonal = float(np.hypot(width, height))
        aspect_ratio = max(width, height) / max(1, min(width, height))
        fill_ratio = area / max(1, width * height)

        if strict:
            looks_like_scratch = (
                aspect_ratio >= 3.2 and diagonal >= minimum_diagonal
            ) or (
                fill_ratio <= 0.14 and diagonal >= minimum_diagonal * 2.0
            )
        else:
            looks_like_scratch = (
                aspect_ratio >= 2.5 and diagonal >= minimum_diagonal
            ) or (
                fill_ratio <= 0.24 and diagonal >= minimum_diagonal * 1.8
            )

        if looks_like_scratch:
            selected[labels == label] = 255

    return selected


def _build_scratch_mask(gray: np.ndarray) -> np.ndarray:
    """세피아 사진의 밝은 균열과 일부 짙은 긁힘을 고신뢰도로 찾습니다."""
    height, width = gray.shape
    longest_side = max(height, width)
    median_size = 7 if min(height, width) >= 7 else 3
    local_median = cv2.medianBlur(gray, median_size)
    bright_residual = cv2.subtract(gray, local_median)
    dark_residual = cv2.subtract(local_median, gray)

    bright_threshold = max(12.0, float(np.percentile(bright_residual, 96.5)))
    dark_threshold = max(22.0, float(np.percentile(dark_residual, 99.0)))
    bright = np.where(bright_residual >= bright_threshold, 255, 0).astype(np.uint8)
    dark = np.where(dark_residual >= dark_threshold, 255, 0).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_kernel)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.bitwise_or(
        _select_scratch_components(bright, longest_side, strict=False),
        _select_scratch_components(dark, longest_side, strict=True),
    )

    _protect_detected_faces(mask, gray)

    # 액자 테두리와 사진 하단의 촬영 문구·날짜는 손상으로 오인하기 쉽습니다.
    border = max(4, round(min(height, width) * 0.015))
    caption_height = max(border, round(height * 0.065))
    mask[:border, :] = 0
    mask[-caption_height:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    dilation_radius = max(1, min(3, round(longest_side / 700)))
    dilation_size = dilation_radius * 2 + 1
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_size, dilation_size),
    )
    mask = cv2.dilate(mask, dilation_kernel, iterations=1)
    if cv2.countNonZero(mask) > mask.size * 0.06:
        # 무늬가 매우 복잡한 사진을 균열로 과검출한 경우 원본 훼손을 우선 막습니다.
        return np.zeros_like(mask)
    return mask


def _repair_damage_and_fading(bgr: np.ndarray) -> np.ndarray:
    """Scratch, noise and faded-contrast restoration for old photos."""
    height, width = bgr.shape[:2]
    longest_side = max(height, width)
    scale = min(1.0, 1800.0 / longest_side)

    if scale < 1.0:
        working = cv2.resize(
            bgr,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = bgr.copy()

    denoised = cv2.fastNlMeansDenoisingColored(
        working,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    scratch_mask = _build_scratch_mask(gray)
    if cv2.countNonZero(scratch_mask) > 0:
        inpaint_radius = max(2, min(4, round(max(gray.shape) / 600)))
        denoised = cv2.inpaint(
            denoised,
            scratch_mask,
            inpaint_radius,
            cv2.INPAINT_TELEA,
        )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    enhanced_lightness = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(
        lightness
    )
    lightness = cv2.addWeighted(lightness, 0.65, enhanced_lightness, 0.35, 0)
    restored = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )

    if scale < 1.0:
        restored = cv2.resize(restored, (width, height), interpolation=cv2.INTER_CUBIC)
    return restored


def _colorize_with_ai(bgr: np.ndarray) -> np.ndarray:
    net = _get_colorization_net()
    scaled = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224), interpolation=cv2.INTER_AREA)
    lightness = resized[:, :, 0] - 50.0

    net.setInput(cv2.dnn.blobFromImage(lightness))
    predicted_ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    predicted_ab = cv2.resize(
        predicted_ab,
        (bgr.shape[1], bgr.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    original_lightness = lab[:, :, 0]
    colorized_lab = np.concatenate(
        (original_lightness[:, :, np.newaxis], predicted_ab),
        axis=2,
    )
    colorized = cv2.cvtColor(colorized_lab, cv2.COLOR_LAB2BGR)
    return np.clip(colorized * 255.0, 0, 255).astype(np.uint8)


def _enhance_details_with_ai(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    longest_side = max(height, width)
    scale = min(1.0, DETAIL_MODEL_MAX_SIDE / longest_side)

    if scale < 1.0:
        model_input = cv2.resize(
            bgr,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        model_input = bgr

    ai_upscaled = _get_detail_net().upsample(model_input)
    reconstructed = cv2.resize(
        ai_upscaled,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.addWeighted(bgr, 0.55, reconstructed, 0.45, 0)


def _restore_photo(
    bgr: np.ndarray,
    restore_damage: bool,
    colorize: bool,
    enhance_details: bool,
) -> tuple[np.ndarray, bool]:
    # 손상 복원부터 디테일 개선까지 순서대로 처리
    with _restore_inference_lock:
        # 사진의 손상과 색바램 복원
        result = _repair_damage_and_fading(bgr) if restore_damage else bgr.copy()
        # 컬러화가 필요한 사진인지 확인
        was_colorized = colorize and _is_grayscale_or_faded(result)

        if was_colorized:
            # 흑백 또는 바랜 사진을 AI로 컬러화
            result = _colorize_with_ai(result)

        if enhance_details:
            # AI로 이미지 디테일 개선
            result = _enhance_details_with_ai(result)

        return result, was_colorized



# Replicate 클라우드 복원 요청
def save_replicate_api_token(api_token: str) -> None:
    """Replicate 토큰을 프로젝트 밖의 사용자 전용 설정 폴더에 저장합니다."""
    normalized_token = api_token.strip()
    if not normalized_token:
        raise ValueError("Replicate API 토큰이 비어 있습니다.")
    if any(character.isspace() for character in normalized_token):
        raise ValueError("Replicate API 토큰에는 공백을 넣을 수 없습니다.")
    if not normalized_token.startswith("r8_") or len(normalized_token) < 20:
        raise ValueError("Replicate API 토큰은 r8_로 시작하는 값을 입력해 주세요.")

    temporary = REPLICATE_TOKEN_FILE.with_suffix(".tmp")
    temporary.write_text(normalized_token, encoding="utf-8")
    temporary.replace(REPLICATE_TOKEN_FILE)
    try:
        os.chmod(REPLICATE_TOKEN_FILE, 0o600)
    except OSError:
        pass


def _get_replicate_api_token() -> str:
    if REPLICATE_TOKEN_FILE.is_file():
        try:
            saved_token = REPLICATE_TOKEN_FILE.read_text(encoding="utf-8").strip()
            if saved_token:
                return saved_token
        except OSError as exc:
            logger.warning("Replicate token file read failed: %s", exc)
    return os.getenv("REPLICATE_API_TOKEN", "").strip()


def _prepare_replicate_input_data_uri(image: Image.Image) -> str:
    """HTTP API에 안전하게 넣을 수 있도록 이미지를 1MB 이하 JPEG로 만듭니다."""
    working = image.convert("RGB")
    longest_side = max(working.size)
    if longest_side > 1536:
        ratio = 1536.0 / longest_side
        working = working.resize(
            (
                max(1, round(working.width * ratio)),
                max(1, round(working.height * ratio)),
            ),
            Image.Resampling.LANCZOS,
        )

    encoded = b""
    for quality in (90, 84, 78, 72, 66):
        output = BytesIO()
        working.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
        encoded = output.getvalue()
        if len(encoded) <= REPLICATE_MAX_INPUT_BYTES:
            break

    while len(encoded) > REPLICATE_MAX_INPUT_BYTES and max(working.size) > 640:
        working = working.resize(
            (
                max(1, round(working.width * 0.85)),
                max(1, round(working.height * 0.85)),
            ),
            Image.Resampling.LANCZOS,
        )
        output = BytesIO()
        working.save(output, format="JPEG", quality=70, optimize=True)
        encoded = output.getvalue()

    if not encoded or len(encoded) > REPLICATE_MAX_INPUT_BYTES:
        raise ReplicateServiceError("Replicate에 보낼 이미지 크기를 줄이지 못했습니다.")

    encoded_text = base64.b64encode(encoded).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_text}"


def _extract_replicate_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            for key in ("detail", "error", "title"):
                value = payload.get(key)
                if value:
                    return str(value)[:500]
    except (ValueError, AttributeError):
        pass

    text = response.text.strip()
    return text[:500] if text else f"HTTP {response.status_code}"


def _extract_replicate_output_url(output) -> str:
    if isinstance(output, str):
        if output.startswith(("https://", "data:image/")):
            return output
        return ""
    if isinstance(output, list):
        for item in output:
            result = _extract_replicate_output_url(item)
            if result:
                return result
        return ""
    if isinstance(output, dict):
        for key in ("url", "image", "output"):
            result = _extract_replicate_output_url(output.get(key))
            if result:
                return result
    return ""


def _read_replicate_output(output_url: str, headers: dict[str, str]) -> bytes:
    if output_url.startswith("data:image/"):
        try:
            metadata, encoded = output_url.split(",", 1)
            if ";base64" not in metadata:
                raise ValueError
            result = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise ReplicateServiceError("Replicate 결과 이미지 형식이 잘못되었습니다.") from exc
    else:
        parsed = urllib.parse.urlparse(output_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not (
            host == "replicate.delivery" or host.endswith(".replicate.delivery")
        ):
            raise ReplicateServiceError("Replicate가 허용되지 않은 결과 주소를 반환했습니다.")

        response = requests.get(output_url, headers=headers, timeout=(10, 90))
        if not response.ok:
            raise ReplicateServiceError(
                "Replicate 결과 이미지를 내려받지 못했습니다: "
                + _extract_replicate_error(response)
            )
        result = response.content

    if not result:
        raise ReplicateServiceError("Replicate 결과 이미지가 비어 있습니다.")
    if len(result) > REPLICATE_MAX_OUTPUT_BYTES:
        raise ReplicateServiceError("Replicate 결과 이미지가 50MB를 초과했습니다.")
    return result


def _normalize_replicate_output(image_bytes: bytes, target_size: tuple[int, int]) -> bytes:
    """복원 결과를 원본 크기의 표준 PNG로 바꿔 편집 캔버스 크기를 보존합니다."""
    try:
        image = Image.open(BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ReplicateServiceError("Replicate 결과를 이미지로 읽을 수 없습니다.") from exc

    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _request_replicate_photo_restoration(
    image: Image.Image,
    api_token: str,
    cancellation_event: threading.Event | None = None,
) -> bytes:
    """Replicate의 FLUX Kontext 사진 복원 모델을 실행하고 PNG를 반환합니다."""
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait=60",
        "Cancel-After": "6m",
    }
    payload = {
        "input": {
            "input_image": _prepare_replicate_input_data_uri(image),
            "output_format": "png",
        }
    }

    response = requests.post(
        REPLICATE_PREDICTION_URL,
        json=payload,
        headers=headers,
        timeout=(15, 75),
    )
    if not response.ok:
        raise ReplicateServiceError(_extract_replicate_error(response))

    try:
        prediction = response.json()
    except ValueError as exc:
        raise ReplicateServiceError("Replicate 응답을 해석할 수 없습니다.") from exc

    prediction_id = str(prediction.get("id", ""))

    def cancel_prediction() -> None:
        if not prediction_id:
            return
        cancel_url = (
            "https://api.replicate.com/v1/predictions/"
            + urllib.parse.quote(prediction_id, safe="-_")
            + "/cancel"
        )
        try:
            requests.post(
                cancel_url,
                headers={"Authorization": f"Bearer {api_token}"},
                timeout=(5, 15),
            )
        except requests.RequestException as exc:
            logger.warning("Replicate prediction cancellation failed: %s", exc)

    def throw_if_cancelled() -> None:
        if cancellation_event is not None and cancellation_event.is_set():
            cancel_prediction()
            raise ReplicateRequestCancelled("클라이언트가 사진 복원 요청을 취소했습니다.")

    deadline = time.monotonic() + REPLICATE_MAX_WAIT_SECONDS
    poll_headers = {"Authorization": f"Bearer {api_token}"}
    while True:
        throw_if_cancelled()
        status = str(prediction.get("status", "")).lower()
        if status == "succeeded":
            break
        if status in {"failed", "canceled"}:
            message = prediction.get("error") or f"예측 상태: {status}"
            raise ReplicateServiceError(str(message)[:500])
        if status not in {"starting", "processing"}:
            raise ReplicateServiceError("Replicate가 알 수 없는 처리 상태를 반환했습니다.")
        if time.monotonic() >= deadline:
            raise ReplicateServiceError("Replicate 사진 복원 시간이 6분을 초과했습니다.")

        if (
            not prediction_id
            or len(prediction_id) > 128
            or not all(character.isalnum() or character in "-_" for character in prediction_id)
        ):
            raise ReplicateServiceError("Replicate 예측 번호가 올바르지 않습니다.")

        time.sleep(2)
        throw_if_cancelled()
        poll_url = (
            "https://api.replicate.com/v1/predictions/"
            + urllib.parse.quote(prediction_id, safe="-_")
        )
        response = requests.get(poll_url, headers=poll_headers, timeout=(10, 30))
        if not response.ok:
            raise ReplicateServiceError(_extract_replicate_error(response))
        try:
            prediction = response.json()
        except ValueError as exc:
            raise ReplicateServiceError("Replicate 처리 상태를 해석할 수 없습니다.") from exc

    output_url = _extract_replicate_output_url(prediction.get("output"))
    if not output_url:
        raise ReplicateServiceError("Replicate가 복원 결과 주소를 반환하지 않았습니다.")

    throw_if_cancelled()
    result_bytes = _read_replicate_output(output_url, poll_headers)
    throw_if_cancelled()
    return _normalize_replicate_output(result_bytes, image.size)



# /restore-photo API (Replicate 먼저, 실패 시 로컬 CPU 복원)
@app.post("/restore-photo")
async def restore_photo(
    request: Request,
    file: UploadFile = File(...),
    restore_damage: bool = Form(True),
    colorize: bool = Form(True),
    enhance_details: bool = Form(True),
):
    # C#에서 보낸 이미지와 복원 옵션 읽기
    data = await _read_uploaded_image(file)

    try:
        pil_image, bgr = _decode_uploaded_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Replicate API 토큰 확인
    replicate_token = _get_replicate_api_token()
    fallback_reason = "token-missing"
    if replicate_token:
        cancellation_event = threading.Event()
        try:
            # Replicate AI를 우선 실행
            replicate_task = asyncio.create_task(
                run_in_threadpool(
                    _request_replicate_photo_restoration,
                    pil_image,
                    replicate_token,
                    cancellation_event,
                )
            )
            while not replicate_task.done():
                if await request.is_disconnected():
                    cancellation_event.set()
                await asyncio.sleep(0.25)
            result_bytes = await replicate_task
            logger.info("Photo restoration succeeded with Replicate")
            return Response(
                content=result_bytes,
                media_type="image/png",
                headers={
                    "X-ImageEditor-AI": "replicate-cloud",
                    "X-ImageEditor-Restoration-Mode": "replicate",
                    "X-ImageEditor-Colorized": "true",
                },
            )
        except (asyncio.CancelledError, ReplicateRequestCancelled):
            cancellation_event.set()
            logger.info("Photo restoration request was cancelled by the client")
            raise HTTPException(status_code=499, detail="사진 복원 요청이 취소되었습니다.")
        except Exception as exc:
            # Replicate 실패 시 로컬 복원으로 전환
            fallback_reason = "replicate-error"
            logger.exception("Replicate restoration failed; using local fallback: %s", exc)

    # 로컬 CPU로 사진 복원 실행
    try:
        result, was_colorized = await run_in_threadpool(
            _restore_photo,
            bgr,
            restore_damage,
            colorize,
            enhance_details,
        )
        result_bytes = await run_in_threadpool(_encode_png, result)
        return Response(
            content=result_bytes,
            media_type="image/png",
            headers={
                "X-ImageEditor-AI": "local-cpu",
                "X-ImageEditor-Restoration-Mode": "local-fallback",
                "X-ImageEditor-Fallback-Reason": fallback_reason,
                "X-ImageEditor-Colorized": str(was_colorized).lower(),
            },
        )
    except ModelDownloadError as exc:
        logger.exception("Photo restoration model download failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Photo restoration failed")
        raise HTTPException(
            status_code=500,
            detail=f"Replicate와 로컬 AI 사진 복원에 모두 실패했습니다: {exc}",
        ) from exc
