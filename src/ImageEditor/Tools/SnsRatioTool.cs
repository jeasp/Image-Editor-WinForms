using System;
using System.Drawing;
using ImageEditor.Core;
using ImageEditor.ImageProcessing;

namespace ImageEditor.Tools
{
    public class SnsRatioTool
    {
        private readonly EditorState state;

        // SNS 변환을 시작했을 때의 기준 이미지
        private Bitmap? baseImage;

        // 이미지 변경 완료 알림
        public event Action? ImageChanged;


        public SnsRatioTool(EditorState state)
        {
            this.state = state;
        }


        // SNS 비율 기능 시작
        public void Start()
        {
            if (state.CurrentImage == null)
                return;

            // 기존 기준 이미지 정리
            baseImage?.Dispose();

            // 현재 이미지를 기준 이미지로 복사
            baseImage = new Bitmap(state.CurrentImage);
        }

        public void EndSession()
        {
            baseImage?.Dispose();
            baseImage = null;
        }


        // 현재 편집 상태를 바꾸지 않는 세션 미리보기 생성
        // 기준 이미지로 SNS 비율 미리보기 생성
        public Bitmap? CreatePreview(int ratioWidth, int ratioHeight)
        {
            if (baseImage == null)
                return null;

            if (ratioWidth <= 0 || ratioHeight <= 0)
                return null;

            Rectangle cropArea = CalculateCenterCrop( baseImage.Size, ratioWidth, ratioHeight );

            // 항상 baseImage 기준으로 Crop
            return BasicImageProcessor.Crop(baseImage, cropArea);
        }


        // 중앙 기준 Crop 영역 계산
        // 목표 비율에 맞는 중앙 자르기 영역 계산
        private Rectangle CalculateCenterCrop(Size imageSize, int ratioWidth,int ratioHeight)
        {
            int imageWidth = imageSize.Width;
            int imageHeight = imageSize.Height;

            // 현재 이미지 비율 계산
            double imageRatio = (double)imageWidth /imageHeight;

            // 목표 SNS 비율 계산
            double targetRatio = (double)ratioWidth / ratioHeight;

            int cropWidth;
            int cropHeight;

            // 원본이 목표보다 가로로 긴 경우
            if (imageRatio > targetRatio)
            {
                // 가로가 길면 좌우 영역 자르기
                cropHeight = imageHeight;

                cropWidth =(int)(imageHeight * targetRatio);
            }
            else
            {
                // 원본이 목표보다 세로로 긴 경우
                // 세로가 길면 위아래 영역 자르기
                cropWidth =imageWidth;

                cropHeight =(int)( imageWidth / targetRatio );
            }

            // 이미지 중앙을 기준으로 자를 위치 계산
            int x = (imageWidth -  cropWidth) / 2;
            int y = (imageHeight - cropHeight) / 2;

            return new Rectangle(x,y, cropWidth, cropHeight );
        }
    }
}
