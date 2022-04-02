using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace ConsciousCar
{
    class Program
    {
        const int imageWidth = 600;
        const int imageHeight = 600;

        private static Mat GenerateMask(IEnumerable<DetectionResult> detections, DetectionLabel labelId)
        {
            var fin_mask = new Mat(imageWidth, imageHeight, MatType.CV_8UC1, Enumerable.Repeat((byte)0, imageWidth * imageHeight).ToArray());
            foreach (var detectionResult in detections)
            {
                if (detectionResult.Label == labelId)
                {
                    Cv2.Add(detectionResult.Mask, fin_mask.Clone(), fin_mask);
                }
            }

            return fin_mask;
        }

        static void Main(string[] args)
        {
            var detector = new Detector(32);

            using (var window1 = new Window("dst image1"))
            using (var window2 = new Window("dst image2"))
            using (var window3 = new Window("dst image3"))
            using (var window4 = new Window("dst image4"))
            using (var capture = new VideoCapture("images\\video.mp4"))
            {
                var frame = new Mat();
                var screenCaptureAgent = new ScreenCaptureAgent(0, 0, imageWidth, imageHeight);
                foreach (var screenCapture in screenCaptureAgent.Capture())
                {
                    try
                    {
                        capture.Read(frame);
                        Cv2.Resize(frame.Clone(), frame, new Size(imageWidth, imageHeight));
                        //var mat = OpenCvSharp.Cv2.
                        //window1.ShowImage(frame);
                        var screen = BitmapConverter.ToMat(screenCapture);
                        var detectionResults = detector.DetectNext(frame);


                        var linesMask = GenerateMask(detectionResults, DetectionLabel.RoadLine);
                        var roadMask = GenerateMask(detectionResults, DetectionLabel.Road);


                        var empty = new Mat(new Size(imageWidth, imageHeight), MatType.CV_8UC3);

                        //Cv2.BitwiseAnd(roadMask.Clone(), linesMask.Clone(), frame, mask: linesMask);

                        window1.ShowImage(screen);
                        window2.ShowImage(linesMask);
                        window3.ShowImage(frame);
                        window4.ShowImage(roadMask);

                        Cv2.WaitKey(1);
                    }
                    catch (Microsoft.ML.OnnxRuntime.OnnxRuntimeException ex)
                    {

                    }
                    catch (Exception _)
                    {
                        screenCaptureAgent.Dispose();
                    }
                }
            }
            Cv2.WaitKey();


            return;
        }
    }
}

