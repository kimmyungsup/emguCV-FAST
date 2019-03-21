using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.IO;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;


namespace emguCVcam
{
    public partial class Form1 : Form
    {

        Emgu.CV.Image<Gray, byte> inputimByte;
        Emgu.CV.Image<Gray, byte> baseimByte = new Emgu.CV.Image<Gray, byte>("test1.jpg");
        Emgu.CV.Image<Gray, byte> baseimByte2 = new Emgu.CV.Image<Gray, byte>("test2.jpg");
        long time;
        int mpoint;
        bool isMatch;
        int state = 0;
        long matchingTime = 0;


        private Capture capture;
        private bool captureInProgress;

        private void ProcessFrame(object sender, EventArgs arg)
        {
            Image<Bgr, Byte> ImageFrame = capture.QueryFrame().Resize(320, 240, Emgu.CV.CvEnum.INTER.CV_INTER_CUBIC);
            inputimByte = ImageFrame.Convert<Gray, Byte>();

            DrawMatche drawmatch = new DrawMatche();

            Stopwatch watch;
            watch = Stopwatch.StartNew();
            watch.Start();

            


            if (state == 0)
            {
                Emgu.CV.Image<Bgr, byte> returnimByte = DrawMatche.Draw(baseimByte, inputimByte, state, out time, out mpoint);
                imageBox2.Image = returnimByte;
                if (mpoint > 7) //3
                {
                    isMatch = true;
                    imageBox1.Image = returnimByte;
                    state = 1;
                    label2.Text = "matching : " + mpoint.ToString();
                    matchingTime += time;
                    label6.Text = matchingTime.ToString();
                }
                
            }

            else
            {
                Emgu.CV.Image<Bgr, byte> returnimByte = DrawMatche.Draw(baseimByte2, inputimByte, state, out time, out mpoint);
                imageBox2.Image = returnimByte;
                if (mpoint > 7) // 3
                {
                    isMatch = true;  
                    imageBox1.Image = returnimByte;
                    state = 0;
                    label2.Text = "matching : " + mpoint.ToString();
                    matchingTime += time;

                    label3.Text = "all match time = " + matchingTime.ToString();
                    matchingTime = 0;
                    isMatch = false;
                    label6.Text = matchingTime.ToString();
                }
                
            }

            watch.Stop();

            if(isMatch) matchingTime += watch.ElapsedMilliseconds;

            label1.Text = "matching : " + mpoint.ToString() + " ->" + isMatch.ToString();
        }


        public Form1()
        {
            InitializeComponent();

            button1.Text = "Start!";
            label1.Text = "matching : ";
            label2.Text = "";
            label3.Text = "";
            label4.Text = "Matched Image";
            label5.Text = "Matching Process";
            label6.Text = "";
            
        }

        private void ReleaseData()
        {
            if (capture != null)
                capture.Dispose();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (capture == null)
            {
                try
                {
                    capture = new Capture();
                }
                catch (NullReferenceException excpt)
                {
                    MessageBox.Show(excpt.Message);
                }
            }

            if (capture != null)
            {
                if (captureInProgress)
                {  

                    state = 0;

                    button1.Text = "Start!"; 
                    Application.Idle -= ProcessFrame;
                }
                else
                { 
                    isMatch = false;
                    button1.Text = "Stop";
                    Application.Idle += ProcessFrame;
                }
                captureInProgress = !captureInProgress;
            }
        }

        private void timer2_Tick(object sender, EventArgs e)
        {
            if(isMatch == true)
            {
                label3.Text = "all matching time = " + matchingTime.ToString();
                if(state == 1)
                {
                    label3.Text = "all matching time = " + matchingTime.ToString();
                    isMatch = false;
                    matchingTime = 0;
                }
                matchingTime++;

            }
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void imageBox1_Click(object sender, EventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {
            
        }



    }



    public class DrawMatche
    {
        /// <summary>
        /// Draw the model image and observed image, the matched features and homography projection.
        /// </summary>
        /// <param name="modelImage">The model image</param>
        /// <param name="observedImage">The observed image</param>
        /// <param name="matchTime">The output total time for computing the homography matrix.</param>
        /// <returns>The model image and observed image, the matched features and homography projection.</returns>
        public static Image<Bgr, Byte> Draw(Image<Gray, Byte> modelImage, Image<Gray, byte> observedImage, int state, out long matchTime, out int p)
        {
            Stopwatch watch;
            HomographyMatrix homography = null;
            

            SURFDetector surfCPU = new SURFDetector(500, false);
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            Matrix<int> indices;

            Matrix<byte> mask;
            int k = 2;
            double uniquenessThreshold = 0.8; 
            if (state == 1) uniquenessThreshold = 0.8; 

            

                //extract features from the object image
                modelKeyPoints = surfCPU.DetectKeyPointsRaw(modelImage, null);
                Matrix<float> modelDescriptors = surfCPU.ComputeDescriptorsRaw(modelImage, null, modelKeyPoints);

                watch = Stopwatch.StartNew();

                // extract features from the observed image
                observedKeyPoints = surfCPU.DetectKeyPointsRaw(observedImage, null);
                Matrix<float> observedDescriptors = surfCPU.ComputeDescriptorsRaw(observedImage, null, observedKeyPoints);
                BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(DistanceType.L2);
                
                matcher.Add(modelDescriptors);

                indices = new Matrix<int>(observedDescriptors.Rows, k);
                using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, k))
                {
                    matcher.KnnMatch(observedDescriptors, indices, dist, k, null);
                    mask = new Matrix<byte>(dist.Rows, 1);
                    mask.SetValue(255);
                    Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
                }

                

                int nonZeroCount = CvInvoke.cvCountNonZero(mask);
                if (nonZeroCount >= 1)
                {
                    nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                    if (nonZeroCount >= 1)
                        homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
                }

                watch.Stop();

                p = mask.ManagedArray.OfType<byte>().ToList().Where(q => q > 0).Count();
            

            //Draw the matched keypoints
            Image<Bgr, Byte> result = Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
               indices, new Bgr(255, 255, 255), new Bgr(255, 255, 255), mask, Features2DToolbox.KeypointDrawType.DEFAULT);

            matchTime = watch.ElapsedMilliseconds;

            

            return result;
        }

       
    }
}
