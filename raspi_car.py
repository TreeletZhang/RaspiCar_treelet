# coding=utf-8
from Adafruit_PWM_Servo_Driver import PWM
import RPi.GPIO as GPIO
import cv2
import time
import keyboard
import random


class RaspiCar:
    # Motor left
    __PWMA = 18
    __AIN1 = 22
    __AIN2 = 27

    # Motor right
    __PWMB = 23
    __BIN1 = 25
    __BIN2 = 24

    # LED
    __GLED = 5
    __RLED = 6

    # Ultrasound
    __TRIG = 20
    __ECHO = 21

    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)

        # Motor
        GPIO.setup(self.__AIN1, GPIO.OUT)
        GPIO.setup(self.__AIN2, GPIO.OUT)
        GPIO.setup(self.__PWMA, GPIO.OUT)

        GPIO.setup(self.__BIN1, GPIO.OUT)
        GPIO.setup(self.__BIN2, GPIO.OUT)
        GPIO.setup(self.__PWMB, GPIO.OUT)

        self.l_motor = GPIO.PWM(self.__PWMA, 100)
        self.l_motor.start(0)

        self.r_motor = GPIO.PWM(self.__PWMB, 100)
        self.r_motor.start(0)

        # LED
        GPIO.setup(self.__GLED, GPIO.OUT)
        GPIO.setup(self.__RLED, GPIO.OUT)

        # Ultrasound
        GPIO.setup(self.__TRIG, GPIO.OUT)
        GPIO.setup(self.__ECHO, GPIO.IN)

        # Camera
        self.cap = cv2.VideoCapture(0)

        # Hat
        self.pwm = PWM(0x40, debug=False)
        self.pwm.setPWMFreq(60)

    def _l_motor_run(self, speed, is_forward=True):
        self.l_motor.ChangeDutyCycle(speed)
        GPIO.output(self.__AIN2, not is_forward)
        GPIO.output(self.__AIN1, is_forward)

    def _r_motor_run(self, speed, is_forward=True):
        self.r_motor.ChangeDutyCycle(speed)
        GPIO.output(self.__BIN2, not is_forward)
        GPIO.output(self.__BIN1, is_forward)

    def _l_motor_stop(self):
        self.l_motor.ChangeDutyCycle(0)
        GPIO.output(self.__AIN2, False)
        GPIO.output(self.__AIN1, False)

    def _r_motor_stop(self):
        self.r_motor.ChangeDutyCycle(0)
        GPIO.output(self.__BIN2, False)
        GPIO.output(self.__BIN1, False)

    # 向前走，传入速度
    def forward(self, speed):
        self._l_motor_run(speed, True)
        self._r_motor_run(speed, True)
    # 向后退
    def backward(self, speed):
        self._l_motor_run(speed, False)
        self._r_motor_run(speed, False)
    # 左转
    # def left(self, speed):
    #     self._l_motor_run(speed - 10, True)
    #     self._r_motor_run(speed + 30, True)
    def left(self, speed):
        self._l_motor_run(speed-20, True)
        self._r_motor_run(speed, True)
    # 右转
    # def right(self, speed):
    #     self._l_motor_run(speed + 30, True)
    #     self._r_motor_run(speed - 10, True)
    def right(self, speed):
        self._l_motor_run(speed, True)
        self._r_motor_run(speed-20, True)
    # 停
    def stop(self):
        self._l_motor_stop()
        self._r_motor_stop()
    # 红灯亮
    def red_led(self):
        GPIO.output(self.__GLED, 1)
        GPIO.output(self.__RLED, 0)
    # 绿灯亮
    def green_led(self):
        GPIO.output(self.__GLED, 0)
        GPIO.output(self.__RLED, 1)
    # 红灯绿灯同时亮
    def red_green_led(self):
        GPIO.output(self.__GLED, 1)
        GPIO.output(self.__GLED, 1)
    # 关灯
    def off_led(self):
        GPIO.output(self.__GLED, 0)
        GPIO.output(self.__GLED, 0)
    # 超声测距，返回距离，单位厘米
    def ultrasound_distance(self):
        GPIO.output(self.__TRIG, 0)
        time.sleep(0.000002)

        GPIO.output(self.__TRIG, 1)
        time.sleep(0.00001)
        GPIO.output(self.__TRIG, 0)

        while GPIO.input(self.__ECHO) == 0:
            pass
        t = time.time()
        while GPIO.input(self.__ECHO) == 1:
            pass

        during = time.time() - t
        return during * 340 / 2 * 100

    def camera_observe(self, scale=1):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])

            return frame / 255.

    def _set_servo_pulse(self, channel, pulse):
        pulseLength = 1000000.0  # 1,000,000 us per second
        pulseLength /= 60.0  # 60 Hz
        pulseLength /= 4096.0  # 12 bits of resolution
        pulse *= 1000.0
        pulse /= (pulseLength * 1.0)
        self.pwm.setPWM(channel, 0, int(pulse))

    def _pwm_write(self, servonum, x):
        y = x / 90.0 + 0.5
        y = max(y, 0.5)
        y = min(y, 2.5)
        self._set_servo_pulse(servonum, y)

    # 超声测距方向，往前就设置90，要往左转10度，就要输入100，
    # 当前角度与上一个角度的差值大于多少，就往左转多少
    # 差值小于零，往右转
    def ultrasound_direction(self, horizon):
        self._pwm_write(0, horizon)

    #相机 horizon方向：小于90度，往右转，大于90度，往左转
    #相机 vertical方向(>=0)：向下转
    def camera_direction(self, horizon, vertical):
        self._pwm_write(1, horizon)
        self._pwm_write(2, vertical)
    # 关闭
    def dispose(self):
        GPIO.cleanup()
        self.cap.release()

        self.ultrasound_direction(90)
        self.camera_direction(90, 0)
        time.sleep(0.2)
        self.pwm.dispose()


if __name__ == "__main__":
    try:
        car = RaspiCar()

        def is_block():
            d = car.ultrasound_distance()
            if d <= 20:
                return True
            return False

        while True:
            if is_block():
                car.red_led()
                car.stop()
            else:
                car.green_led()
                a = input()
                if a=='w':
                    car.forward(20)
                elif a=="s":
                    car.backward(20)
                elif a=='a':
                    car.left(30)
                elif a=='d':
                    car.right(30)
                else:
                    car.stop()


        # while True:
        #     if is_block():
        #         car.red_led()
        #         car.stop()
        #     else:
        #         car.green_led()
        #         if keyboard.is_pressed('w'):
        #             car.forward(20)
        #         elif keyboard.is_pressed('s'):
        #             car.backward(20)
        #         elif keyboard.is_pressed('a'):
        #             car.left(10)
        #         elif keyboard.is_pressed('d'):
        #             car.right(10)
        #         elif keyboard.is_pressed('q'):
        #             car.stop()

            # d = car.ultrasound_distance()
            # print(d)

        # for i in range(20):
        #     car.camera_direction(90, i - 1)
        #     time.sleep(1)
        #     frame = car.camera_observe(0.1)
        #     print(frame.shape)

        # while True:
        #     a=input()
        #     car.ultrasound_direction(float(a))

        car.dispose()
    except KeyboardInterrupt:
        car.dispose()
