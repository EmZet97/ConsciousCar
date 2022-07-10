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
                if (detectionResult.Label != labelId) continue;
                //Cv2.Add(detectionResult.Mask, fin_mask.Clone(), fin_mask);
                Cv2.BitwiseOr(fin_mask.Clone(), detectionResult.Mask, fin_mask);
                //Cv2.Rectangle(fin_mask,
                //    new OpenCvSharp.Point(detectionResult.Box.Point1.X, detectionResult.Box.Point1.Y),
                //    new OpenCvSharp.Point(detectionResult.Box.Point2.X, detectionResult.Box.Point2.Y),
                //    new Scalar(128),
                //    thickness: 2);
                //yield return fin_mask;
            }

            yield return fin_mask;
        }

        static void Main(string[] args)
        {
            var detector = new Detector(228, 0.6f);
            var imageProcessor = new ImageProcessor(imageHeight, imageWidth);

            using (var window1 = new Window("dst image1"))
            using (var window2 = new Window("dst image2"))
            //using (var window3 = new Window("dst image3"))
            //using (var window4 = new Window("dst image4"))
            //using (var capture = new VideoCapture("images\\video.mp4"))
            {
                //Cv2.CreateTrackbar("test1", "dst image2", 500);
                //Cv2.CreateTrackbar("test2", "dst image2", 500);
                //Cv2.CreateTrackbar("test3", "dst image2", 500);
                //Cv2.CreateTrackbar("test4", "dst image2", 500);
                //Cv2.CreateTrackbar("test5", "dst image2", 500);
                //Cv2.CreateTrackbar("test6", "dst image2", 1000);
                //var frame = new Mat();
                var screenCaptureAgent = new ScreenCaptureAgent(0, 0, imageWidth, imageHeight);
                //var controller = new GameControllAgent();
                int counter = 0;
                foreach (var screenCapture in screenCaptureAgent.Capture())
                {
                    try
                    {
                        //capture.Read(frame);
                        counter++;

                        //if (counter % 10 != 0)
                        //    controller.MoveForward();
                        //else
                        //    controller.Stop();

                        //Cv2.Resize(frame.Clone(), frame, new Size(imageWidth, imageHeight));

                        var screen_capture = BitmapConverter.ToMat(screenCapture);
                        var screen = new Mat();
                        Cv2.CvtColor(screen_capture, screen, ColorConversionCodes.BGRA2BGR);

                        var detectionResults = detector.DetectNext(screen.Clone());


                        var detection1 = GenerateMask(detectionResults, DetectionLabel.Car);

                        //var detection2 = GenerateMask(detectionResults, DetectionLabel.Road);
                        //var detection3 = GenerateMask(detectionResults, DetectionLabel.Sidewalk);


                        var empty = new Mat(new Size(imageWidth, imageHeight), MatType.CV_8UC3);

                        //Cv2.BitwiseAnd(roadMask.Clone(), linesMask.Clone(), frame, mask: linesMask);

                        window1.ShowImage(screen);
                        var res = imageProcessor.GenerateNext(screen, detectionResults);
                        window2.ShowImage(res);
                        //foreach (var det in detection1)
                        //{
                        //    var fin = screen.Clone();
                        //    var d = det.Clone();
                        //    Cv2.CvtColor(screen.Clone(), fin, ColorConversionCodes.BGR2GRAY);
                        //    Cv2.BitwiseAnd(d.Clone(), fin.Clone(), fin);//, mask: det);
                        //    var t4 = Cv2.GetTrackbarPos("test4", "dst image2") + 1;
                        //    Cv2.Blur(fin.Clone(), fin, new Size(t4, t4));
                        //    var t1 = Cv2.GetTrackbarPos("test1", "dst image2");
                        //    var t2 = Cv2.GetTrackbarPos("test2", "dst image2");
                        //    var t3 = Cv2.GetTrackbarPos("test3", "dst image2");
                        //    var t5 = Cv2.GetTrackbarPos("test5", "dst image2");
                        //    var t6 = Cv2.GetTrackbarPos("test6", "dst image2") * -1;
                        //    Cv2.AddWeighted(fin.Clone(), t5, fin.Clone(), 0, t6, fin);
                        //    window3.ShowImage(fin);

                        //    Cv2.Canny(fin.Clone(), fin, t1, t2, apertureSize: 3, L2gradient: false);
                        //    var lines = Cv2.HoughLines(fin.Clone(), 1, Math.PI / 180, t3);
                        //    //fin = new Mat(imageHeight, imageWidth, MatType.CV_8UC1, Enumerable.Repeat((byte)0, imageWidth * imageHeight).ToArray());
                        //    foreach (var line in lines)
                        //    {
                        //        var rho = line.Rho;
                        //        var theta = line.Theta;
                        //        var a = MathF.Cos(theta);
                        //        var b = MathF.Sin(theta);
                        //        var x0 = a * rho;
                        //        var y0 = b * rho;
                        //        var pt1 = ((int)(x0 + 1000 * (-b)), (int)(y0 + 1000 * (a)));
                        //        var pt2 = ((int)(x0 - 1000 * (-b)), (int)(y0 - 1000 * (a)));

                        //        var p1 = line.ToSegmentPoint(1).P1;
                        //        var p2 = line.ToSegmentPoint(1).P2;
                        //        Cv2.Line(fin, pt1.Item1, pt1.Item2, pt2.Item1, pt2.Item2, new Scalar(128), 1);
                        //    }
                        //    window2.ShowImage(fin);
                        //    //Cv2.WaitKey(500);
                        //    //break;
                        //}
                        //window3.ShowImage(detection2);
                        //window4.ShowImage(detection3);

                        Cv2.WaitKey(10);

                    }
                    catch (Microsoft.ML.OnnxRuntime.OnnxRuntimeException ex)
                    {

                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                        //screenCaptureAgent.Dispose();
                    }
                }
            }
            Cv2.WaitKey();


            return;
        }
    }
}

