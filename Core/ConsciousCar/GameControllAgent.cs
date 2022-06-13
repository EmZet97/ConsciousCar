using Nefarius.ViGEm.Client;
using Nefarius.ViGEm.Client.Targets;
using Nefarius.ViGEm.Client.Targets.Xbox360;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsciousCar
{
    internal class GameControllAgent
    {
        private readonly IXbox360Controller controller;

        public GameControllAgent()
        {
            var client = new ViGEmClient();

            controller = client.CreateXbox360Controller();

            controller.Connect();
        }


        public void MoveForward()
        {
            controller.SetButtonState(Xbox360Button.RightShoulder, false);

            controller.SetAxisValue(Xbox360Axis.LeftThumbY, short.MaxValue);
            controller.SetSliderValue(Xbox360Slider.RightTrigger, byte.MaxValue / 3);
            controller.SetSliderValue(Xbox360Slider.LeftTrigger, 0);

            controller.SetAxisValue(Xbox360Axis.RightThumbY, short.MinValue / 3);
        }

        public void MoveBackward()
        {
            controller.SetButtonState(Xbox360Button.RightShoulder, false);

            controller.SetAxisValue(Xbox360Axis.LeftThumbY, short.MinValue);
            controller.SetSliderValue(Xbox360Slider.RightTrigger, 0);
            controller.SetSliderValue(Xbox360Slider.LeftTrigger, byte.MaxValue / 3);
        }

        public void Stop()
        {
            controller.SetAxisValue(Xbox360Axis.LeftThumbX, 0);
            controller.SetButtonState(Xbox360Button.RightShoulder, true);
        }
    }
}
