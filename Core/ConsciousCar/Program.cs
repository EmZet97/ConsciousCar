using System;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Extensions;

namespace ConsciousCar
{
    class Program
    {
        const int imageWidth = 800;
        const int imageHeight = 600;

        private static IEnumerable<Mat> GenerateMask(IEnumerable<DetectionResult> detections, DetectionLabel labelId)
        {
            var fin_mask = new Mat(imageHeight, imageWidth, MatType.CV_8UC1, Enumerable.Repeat((byte)0, imageWidth * imageHeight).ToArray());
            foreach (var detectionResult in detections)
            {
                Cv2.Add(detectionResult.Mask, fin_mask.Clone(), fin_mask);

                yield return fin_mask;
            }

            //return fin_mask;
        }

        static void Main(string[] args)
        {
            var detector = new Detector(50);

            using (var window1 = new Window("dst image1"))
            using (var window2 = new Window("dst image2"))
            //using (var window3 = new Window("dst image3"))
            //using (var window4 = new Window("dst image4"))
            using (var capture = new VideoCapture("images\\video.mp4"))
            {
                var frame = new Mat();
                var screenCaptureAgent = new ScreenCaptureAgent(0, 0, imageWidth, imageHeight);
                var controller = new GameControllAgent();
                int counter = 0;
                foreach (var screenCapture in screenCaptureAgent.Capture())
                {
                    try
                    {
                        capture.Read(frame);
                        counter++;

                        if (counter % 10 != 0)
                            controller.MoveForward();
                        else
                            controller.Stop();

                        Cv2.Resize(frame.Clone(), frame, new Size(imageWidth, imageHeight));

                        var screen_capture = BitmapConverter.ToMat(screenCapture);
                        var screen = new Mat();
                        Cv2.CvtColor(screen_capture, screen, ColorConversionCodes.BGRA2BGR);

                        var detectionResults = detector.DetectNext(screen.Clone());


                        var detection1 = GenerateMask(detectionResults, DetectionLabel.Other);

                        //var detection2 = GenerateMask(detectionResults, DetectionLabel.Road);
                        //var detection3 = GenerateMask(detectionResults, DetectionLabel.Sidewalk);


                        var empty = new Mat(new Size(imageWidth, imageHeight), MatType.CV_8UC3);

                        //Cv2.BitwiseAnd(roadMask.Clone(), linesMask.Clone(), frame, mask: linesMask);

                        window1.ShowImage(screen);
                        foreach (var det in detection1)
                        {
                            window2.ShowImage(det);
                            Cv2.WaitKey(100);

                        }
                        //window3.ShowImage(detection2);
                        //window4.ShowImage(detection3);


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

