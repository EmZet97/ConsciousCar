using System;
using System.Linq;
using OpenCvSharp;

namespace ConsciousCar
{
    class Program
    {
        const int imageWidth = 600;
        const int imageHeight = 600;

        static void Main(string[] args)
        {
            var detector = new RoadDetector();

            using (var window1 = new Window("dst image1"))
            using (var window2 = new Window("dst image2"))
            using (var window3 = new Window("dst image3"))
            using (var capture = new VideoCapture("images\\video.mp4"))
            {
                var frame = new Mat();
                var run = true;
                while (run)
                {
                    try
                    {
                        capture.Read(frame);
                        Cv2.Resize(frame.Clone(), frame, new Size(imageWidth, imageHeight));
                        var masks = detector.DetectNext(frame);
                        window1.ShowImage(frame);

                        var fin_mask = new Mat(imageWidth, imageHeight, MatType.CV_8UC1, Enumerable.Repeat((byte)0, imageWidth * imageHeight).ToArray());
                        foreach (var mask in masks)
                        {
                            var gray = new Mat(imageWidth, imageHeight, MatType.CV_8UC1, mask.Select(p => (byte)(p * 255)).Select(p => (byte)(p > 128 ? 0 : 255)).ToArray());
                            Cv2.Add(gray, fin_mask.Clone(), fin_mask);
                            break;
                        }

                        window2.ShowImage(fin_mask);

                        var empty = new Mat(new Size(imageWidth, imageHeight), MatType.CV_8UC3);

                        Cv2.BitwiseAnd(frame.Clone(), empty, frame, mask: fin_mask);
                        window3.ShowImage(frame);

                        Cv2.WaitKey(1);
                    }
                    catch(Microsoft.ML.OnnxRuntime.OnnxRuntimeException ex)
                    {

                    }
                    catch(Exception ex)
                    {
                        run = false;
                    }
                    }
            }
                Cv2.WaitKey();
            
                
            return;
            //using var src = new Mat("images\\road.jpg", ImreadModes.Grayscale);
            //using var dst = new Mat();
            //using var blr = new Mat();
            //using var blr2 = new Mat();
            //using var blr3 = new Mat();

            ////Cv2.Canny(src, dst, 50, 300);
            //Cv2.Blur(src, blr, new Size(3, 2));
            //Cv2.BilateralFilter(src, blr2, 9, 0, 50);
            //Cv2.MedianBlur(src, blr3, 5);
            //Cv2.Canny(blr, dst, 50, 300);



            //using (new Window("src image", src))
            //using (new Window("blr image", blr))
            //using (new Window("blr2 image", blr2))
            //using (new Window("blr3 image", blr3))
            //using (new Window("dst image", dst))
            //{
            //    Cv2.WaitKey();
            //}

            int val1 = 0;
            int val2 = 0;
            int val3 = 0;
            Cv2.NamedWindow("Test");
            Cv2.CreateTrackbar("Contrast", "Test", ref val1, 255);
            Cv2.CreateTrackbar("Contrast2", "Test", ref val2, 255);
            Cv2.CreateTrackbar("Contrast3", "Test", ref val3, 255);

            using (var capture = new VideoCapture("images\\video.mp4"))
            using (var window = new Window("window"))
            using (var win = new Window("win"))
            using (var window1 = new Window("dst image1"))
            using (var window2 = new Window("dst image2"))
            using (var window3 = new Window("dst image3"))
            {
                var run = true;
                while (run)
                {
                    var frame = new Mat();
                    var nframe = new Mat();
                    try
                    {
                        capture.Read(frame);
                        Cv2.Resize(frame.Clone(), frame, new Size(600, 400));
                        //Cv2.CvtColor(nframe.Clone(), nframe, ColorConversionCodes.RGB2GRAY);
                        //Cv2.CvtColor(nframe.Clone(), nframe, ColorConversionCodes.RGB2YCrCb);
                        Cv2.CvtColor(frame.Clone(), nframe, ColorConversionCodes.RGB2HSV);


                        //Cv2.CvtColor(nframe.Clone(), nframe, ColorConversionCodes.RGB2Lab);
                        var arr = new Mat[] { };
                        Cv2.Split(nframe.Clone(), out arr);
                        //Cv2.mer
                        Cv2.BilateralFilter(arr[1], nframe, val1, val2, val3);
                        //Cv2.MedianBlur(arr[1], nframe, 5);
                        //Cv2.GaussianBlur(nframe.Clone(), nframe, new Size(5, 5), 0);
                        //Cv2.GaussianBlur(nframe.Clone(), nframe, new Size(5, 5), 0);
                        //Cv2.GaussianBlur(nframe.Clone(), nframe, new Size(5, 5), 0);
                        //Cv2.MedianBlur(nframe.Clone(), nframe, 5);
                        //Cv2.Canny(nframe.Clone(), nframe, 50, 300);
                        window.ShowImage(frame);
                        win.ShowImage(nframe);
                        window1.ShowImage(arr[0]);
                        window2.ShowImage(arr[1]);
                        window3.ShowImage(arr[2]);
                        Cv2.WaitKey(50);
                    }
                    catch (Exception _)
                    {
                        run = false;
                    }
                }
            }

            Cv2.WaitKey();
        }
    }
}

