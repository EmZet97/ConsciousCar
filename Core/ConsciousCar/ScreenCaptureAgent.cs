using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;

namespace ConsciousCar
{
    internal class ScreenCaptureAgent
    {
        public int ScreenX { get; }
        public int ScreenY { get; }
        public int Width { get; }
        public int Height { get; }

        private bool _disposed = false;

        public ScreenCaptureAgent(int screenX, int screenY, int width, int height)
        {
            ScreenX = screenX;
            ScreenY = screenY;
            Width = width;
            Height = height;
        }

        public IEnumerable<Bitmap> Capture()
        {
            var captureBmp = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);
            using var captureGraphic = Graphics.FromImage(captureBmp);

            while (!_disposed)
            {
                captureGraphic.CopyFromScreen(ScreenX, ScreenY, 0, 0, captureBmp.Size);
                yield return captureBmp;
            }
        }

        public void Dispose()
        {
            _disposed = true;
        }
    }
}
