#include <Arduino_LSM6DS3.h>
#include "Kalman.h"
#include <math.h>
#include <Servo.h>
Servo servo;
Kalman kalman;

//change in time millis
long currMillis, prevMillis;
int cntSpeedCntrl;

//pin numbers for motor driver
const int dirM1 = 10; // for motor 1 direction pin
const int speedM1 = 9; // for motor 1 speed pin
const int dirM2 = 12; // for motor 2 direction pin
const int speedM2 = 11;// for motor 2 speed pin

//pin numbers for encoder feedback
const int encM1A = 2;
const int encM2A = 13;

//encoder feedback variables
int encM1PulseNumSpeed;
int encM2PulseNumSpeed;
static unsigned long encM1CntA;
static unsigned long encM2CntA;

//motor's pwm values
double pwm_M1;
double pwm_M2;

//accel and gyro readings and IMU filter params
float ax, ay, az, anglex;
float gx, gy, gz, gyrox, gyroz;
float dt, Q_angle, Q_gyro, R_angle, C_0, K1;
float kalmanrate_new = 0, kalmanrate_old = 0, kalmanrate_filtered;

//control system variables
double kp_balance = 47.5, kd_balance = 0.6; //kp balance was 37.5 and kd balance was 0.6 //On nov 26 2022, kpb = 47.5 and kdb = 0.6
double kp_speed = 7.5, ki_speed = 0.08; //kp speed was 7.5 and ki speed was 0.08 //On nov 26 2022, kps = 7.5 and kis = 0.08
double kp_rotation = 1, kd_rotation = 0;
float kp_pathTracking = 1.5;
float ki_pathTracking = 0.000; //was 0.001
float kd_pathTracking =  0.1; //was 0.01
float kp_Precise = 1;
float lineTrackingIntegral = 0;
float lineTrackingDiff = 0;
//speed and rotation control and filtering variables
int setCarSpeed = 0, setRotationSpeed_M1 = 0, setRotationSpeed_M2 = 0;
int setCarSpeedTemp = 0;
double x = 0.0;
double speedFilter;
double prevSpeedFilter;
double carSpeedInt;
int count;
double setAngle; //just added 9/16/2022 for Prototypes Angle
enum prototypeStates{
  Start,
  Forward,
  Turn,
  PickUp,
  PutDown,
  Idle
};
double balance_control_output, speed_control_output, rotation_control_output,path_tracking_control_output_M1,path_tracking_control_output_M2;
double wheelBiasM1,wheelBiasM2;
//Stuff for Line Following
const int sensorsRep[6] = {14,15,16,17,6,21};
float sensorWeights[6] = {-12.0,-12.0,-10.0,10.0,12.0,12.0};
float constants[6] = {1,2,3,4,5,6};
boolean allBlack = true;
boolean allWhite = true;
float prevPIDOutput;
float prevLineTrackingError;
float lineTrackingError;
float sensorMin[6] = {0,0,0,0,0,0};
float sensorMax[6] = {604.0,569.0,648.0,596.0,654.0,627.0}; //{692.0,628.0,656.0,613.0,655.0,712.0};  NEW {624,581,651,610,652,635}; OLD
float sensors[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
float WeightedSum = 0;
float error;
float calibratedSum;
float integral =0;
float derivative =0;
float old_prop_WeigSum = 0;

int pathTrackingFlag =0;
int ArmDown = 0;
int servoAngle = 10;
int servoIncrement = 1;
int actuatorCount = 0;
int actuatorCountIncrement = 1;
int openClosed = 0; //if 0, arm is open, if 1, arm is closed
int returnFlag = 0;
int done = 0;
//Ultrasonic Code
int risingEdge; //For Ultrasonic
int fallingEdge; //For Ultrasonic
long cm = 100;
int ultrasonic_Counter = 0;
int ultrasonic_Edge = 0; 
int previousTime = 0;
int ultra_flag = 1;

///180 turn stuff
int turnFlag = 0;
void setup() {
  Serial.begin(9600);

  if (!IMU.begin()) {
    //Serial.println("Failed to initialize IMU!");
    while (1);
  }

  //Ultrasonic setup
  pinMode(7, OUTPUT);//define arduino pin Trigger
  pinMode(3, INPUT);//define arduino pin Echo  
  
  //Servo
  servo.attach(5);
  servo.write(servoAngle);
  //Actuator
  pinMode(8,OUTPUT);
  pinMode(4,OUTPUT);

  //motor 1 pin assignment
  pinMode(dirM1, OUTPUT);
  pinMode(speedM1, OUTPUT);

  //motor 2 pin assignment
  pinMode(dirM2, OUTPUT);
  pinMode(speedM2, OUTPUT);
            //pinMode(3, OUTPUT);//define arduino pin Trigger was 8
            //pinMode(7, INPUT);//define arduino pin Echo was 3


  //IR Array
  pinMode(14,INPUT);
  pinMode(15,INPUT);
  pinMode(16,INPUT);
  pinMode(17,INPUT);
  pinMode(20,INPUT);
  pinMode(21,INPUT);
  //
  digitalWrite(8,HIGH);
  digitalWrite(4,LOW);
  delay(5000);
  //attach interrupts to encoder's M1 and M2 inputs
  attachInterrupt(digitalPinToInterrupt(encM1A), encM1CntA_ISR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encM2A), encM2CntA_ISR, CHANGE);
  //attachInterrupt(digitalPinToInterrupt(3), echo_ISR, CHANGE);
  dt = 0.008, Q_angle = 0.001, Q_gyro = 0.005, R_angle = 0.5, C_0 = 1, K1 = 0.05; 
}
  prototypeStates currState = Start;
void loop() {
  currMillis = millis();
  if(currMillis - prevMillis > 5){
      prevMillis = currMillis;
    switch(currState){
      case Start:
          setCarSpeed = 0;
          setRotationSpeed_M1 = 0; //changed from 50  11/22/2022
          setRotationSpeed_M2 = 0;
          setAngle = -0.3; //was -0.3
          if(done == 1){ //You are done with the task
              currState = Idle;
          }
          else{ //You are not done
            if(count <= 1000){ //Balances for 5 Seconds, checks for next state
              currState = Start;
            }
            else{
              currState = Forward; //was Forward, changed on 9/9/2022
            }
          }
      break;
      case Forward:
        setAngle = -0.3; //changed from 1.0 or 1.5
        setCarSpeed = 10;
        if(returnFlag == 0){ //We are not returning yet
          pathTrackingFlag = 1;
        }
        else{ //We are returning but we only want to activate pathTracking 
        }
        pathTrackingFlag = 1; //was 1
        if(ultra_flag == 1){ //Using Distance Measurement
          getDistance();
          if(cm > 20){ //Not within 11 inches of the payload
            currState = Forward;
          }
          else{ //Is within 8 inches of the payload
            count = 0;
            cm = 100;
            setCarSpeed = 0;
            ArmDown = 1;
            pathTrackingFlag = 0;
            currState = PickUp; //was PickUp 9/7/2022
          }
        }
        else{ //Looking for the Black Line i.e ultra_flag = 0
            if(analogRead(14) > 400 && analogRead(21) > 400){
            setCarSpeed = 0;
            ArmDown = 2;
            pathTrackingFlag = 0;
            currState = PutDown;
          }
          else{
            currState = Forward;
          }
        }
      break;
      case Turn:
        ArmDown = 0;
        if(count <= 3075){ //Left for 5 seconds at speed = 2000
          setCarSpeed = 0;
          setRotationSpeed_M1 = 15; //changed from 50  11/22/2022
          setRotationSpeed_M2 = 15;
        }
        else{
          turnFlag = 0;
          count = 0;
          setCarSpeed = 0;
          setRotationSpeed_M1 = 0; //changed from 50  11/22/2022
          setRotationSpeed_M2 = 0;
          ultra_flag = 0;
          returnFlag = 0;
          currState = Start;
        }
      break;
      case PickUp:
        if(ArmDown == 1){ //Stop for 5 seconds at speed = 2000
          setCarSpeed = 0;
          setRotationSpeed_M1 = 0; //changed from 50  11/22/2022
          setRotationSpeed_M2 = 0;
        }
        else{
          count = 0;
          pathTrackingFlag = 0;
          //turnFlag = 1;
          returnFlag = 0; // was 0?
          currState = Forward;
        }
        ultra_flag = 0;
      break;
      case PutDown:
          pathTrackingFlag = 0;
          setCarSpeed = 0;
        if(ArmDown == 2){ //Stop for 5 seconds at speed = 2000
          setRotationSpeed_M1 = 0; //changed from 50  11/22/2022
          setRotationSpeed_M2 = 0;
        }
        else{
          count = 0;
          pathTrackingFlag = 0;
          done = 1;
          currState = Start;
        }
        ultra_flag = 0;
      break;
      case Idle:
        currState = Idle;
      break;
      default:
        currState = Start;
      break;
    }
    count++;
    //sample feedback from plant sensors
    sampleEncoders();
    sampleIMU();
    //filter IMU readings with kalman and FIR on gyrox
    dt = 0.008, Q_angle = 0.001, Q_gyro = 0.005, R_angle = 0.5, C_0 = 1, K1 = 0.05;
    kalman.getAngle(anglex, gyrox, dt);
    filter_kalmanrate();
    if(ArmDown == 1){ //Pick Up Routine
      if(actuatorCount == 0){ //Lower Actuator
        servoIncrement = 0;
        digitalWrite(8,HIGH);
        digitalWrite(4,LOW);
      }
      else if(actuatorCount == 1000){ //Close Arm to 80
        if(openClosed == 0){ //Arm is open
          servoIncrement = 1;
          openClosed = 1;
        }
        else{ //Arm is closed
          servoIncrement = -1;
          openClosed = 0;
        }
      }
      else if(actuatorCount == 1300){ //Raise Actuator Arrives
        digitalWrite(8,LOW);
        digitalWrite(4 ,HIGH);
      }
      else if(actuatorCount == 2600){ //Reset
        actuatorCount = 0;
        servoIncrement = 0;
        ArmDown = 0;
      }
      if(servoAngle == 80){ //If closed
        servoIncrement = 0;
        servoAngle = 79;
      }
      if(servoAngle == 9){  
        servoIncrement = 0;
        servoAngle = 10;
      }
      actuatorCount = actuatorCount + actuatorCountIncrement;
      servo.write(servoAngle);
      servoAngle = servoAngle + servoIncrement;
    }
    else if (ArmDown == 2){
      if(actuatorCount == 1){ //Lower Actuator
        servoIncrement = 0;
        digitalWrite(8,HIGH);
        digitalWrite(4,LOW);
      }
      else if(actuatorCount == 201){ //Close Arm to 80
        if(servoAngle > 70){
          if(openClosed == 0){ //Arm is open
            servoIncrement = 1;
            openClosed = 1;
          }
          else{ //Arm is closed
            servoIncrement = 0;
            servoAngle = 30;
            openClosed = 0;
          }
        }
        else{
          
        }
      }
      else if(actuatorCount == 602){ //Reset
        actuatorCount = 0;
        servoIncrement = 0;
        ArmDown = 3;
      }
      if(servoAngle == 80){ //If closed
        servoIncrement = 0;
        servoAngle = 79;
      }
      if(servoAngle == 9){  
        servoIncrement = 0;
        servoAngle = 10;
      }
      actuatorCount = actuatorCount + actuatorCountIncrement;
      servo.write(servoAngle);
      servoAngle = servoAngle + servoIncrement;
    }
    //perform balance control calculation roughly every 5 ms
    balanceControlPD(kalman.angle, setAngle, kalmanrate_filtered, 0.0);
    cntSpeedCntrl++;
    if (cntSpeedCntrl == 5) { //was 5
      cntSpeedCntrl = 0;
      if(turnFlag == 1){
        preciseTurn();
      }
      //sample and filter current car speed
      sample_filterSpeed();  
      //perform balance control calculation roughly every 25 ms
      speedControlPI(speedFilter, setCarSpeed);
      rotationControlPD();
      if(pathTrackingFlag == 1){ //When one, include the path tracking control
        pathTrackingControlPID();
      } 
    }
    pwm_M1 = balance_control_output + (-speed_control_output - path_tracking_control_output_M1) + setRotationSpeed_M1 - wheelBiasM1 + 0.5*gyroz;
    pwm_M2 = balance_control_output + (-speed_control_output - path_tracking_control_output_M2) - setRotationSpeed_M2 - wheelBiasM2 + 0.5*gyroz;
    pwm_M1 = constrain(pwm_M1, -255, 255);
    pwm_M2 = constrain(pwm_M2, -255, 255);
    actuateMotors(pwm_M1, pwm_M2); 
  }
}
  
void balanceControlPD(double feedbackAngle, double refAngle, double feedbackAngleVel, double refAngleVel){
  //put angle error through PD controller
  balance_control_output = kp_balance * (feedbackAngle - refAngle) + kd_balance * (feedbackAngleVel - refAngleVel);
}

void speedControlPI(double feedbackSpeed, int refSpeed){
    carSpeedInt += feedbackSpeed;
    carSpeedInt += -refSpeed; //was refSpeed changed to setCarSpeedTemp but changing it back to refSpeed 9/7/2022
    carSpeedInt = constrain(carSpeedInt, -3000, 3000);
    speed_control_output = -kp_speed * (speedFilter) - ki_speed * carSpeedInt;
}

void rotationControlPD(){
  //rotation_control_output = kp_rotation * setRotationSpeed + kd_rotation * gyroz;
}

void getDistance(){
  ultrasonic_Counter++;
  if(ultrasonic_Counter == 10){ //Count has reached 50 ms
    ultrasonic_Counter = 0;
    ultrasonic_Edge = 0;
    //Serial.println("In Get Distance");
    digitalWrite(7, LOW);
    delayMicroseconds(4);
    digitalWrite(7, HIGH);
    delayMicroseconds(10);
    digitalWrite(7, LOW); 
    attachInterrupt(digitalPinToInterrupt(3), echoInterrupt, RISING);
  }
}

void echoInterrupt(){
  if(ultrasonic_Edge == 0){ //Expect Rising Edge
    previousTime = micros();
    attachInterrupt(digitalPinToInterrupt(3),echoInterrupt,FALLING);
    ultrasonic_Edge = 1;
  }
  else if (ultrasonic_Edge == 1){ //Expect Falling Edge
    cm = (micros() -previousTime)*0.017;
    Serial.println(String(cm));
    ultrasonic_Edge = 2;
  }
}

void pathTrackingControlPID(){
  WeightedSum = 0;
  for(int i = 0; i <= 5; i++){
    float meas = analogRead(sensorsRep[i]) - sensorMin[i];
    float divide = sensorMax[i]-sensorMin[i];
    sensors[i] = (meas/divide);
    WeightedSum += sensors[i] * sensorWeights[i];
  }
  error = WeightedSum - (0.75); //65 was calibratedSum was 74.3ish
  integral += error;
  derivative =  error - old_prop_WeigSum;
  old_prop_WeigSum = error;
  float pathTrackingControlOutputPID = kp_pathTracking*(error) + ki_pathTracking*integral + kd_pathTracking*derivative;
  if(pathTrackingControlOutputPID < 0){ //Sensors 1-3 are being impacted, drive the right side of the bot
      path_tracking_control_output_M2 =  (-pathTrackingControlOutputPID);
      path_tracking_control_output_M1 = 0;
  }
  else{ //Sensors 4-6 are being impacted, drive the left side of the bot
      path_tracking_control_output_M2 = 0;
      path_tracking_control_output_M1 = pathTrackingControlOutputPID;
  }
}
void sampleIMU(){
  //read current gyro and angle (feedback)
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    anglex = atan2(ay, az) * 57.3 ;
  }

  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
    //gyrox = (gx - 128.1) / 131;
    gyrox = gx;
    gyroz = gz;
  }
}

void filter_kalmanrate(){
  kalmanrate_old = kalmanrate_new;
  kalmanrate_new = kalman.rate;
  kalmanrate_filtered = 0.7*kalmanrate_old + 0.3*kalmanrate_new;
}

void sample_filterSpeed(){
  //calculate current car speed
  double currCarSpeed = (encM1PulseNumSpeed + encM2PulseNumSpeed) * 0.5;
  encM1PulseNumSpeed = 0;
  encM2PulseNumSpeed = 0;

  //filter the car speed value
  speedFilter = (prevSpeedFilter * 0.7 + currCarSpeed * 0.3);
  prevSpeedFilter = speedFilter;
}

void sampleEncoders(){
  //get encoder speed feedback readings
  if (pwm_M1 < 0) {
    encM1PulseNumSpeed += -encM1CntA;
  }
  else {
    encM1PulseNumSpeed += encM1CntA;
  }

  if (pwm_M2 < 0) {
    encM2PulseNumSpeed += -encM2CntA;
  }
  else {
    encM2PulseNumSpeed += encM2CntA;
  }
  encM1CntA = 0;
  encM2CntA = 0;
}
void actuateMotors(double pwmM1, double pwmM2){
  if(pwmM2 < 0){
    digitalWrite(dirM2, HIGH);// set direction
  }
  else{
    digitalWrite(dirM2, LOW);// set direction
  }
  if(pwmM1 < 0){
    digitalWrite(dirM1, HIGH);// set direction
  }
  else{
    digitalWrite(dirM1, LOW);// set direction
  }
  
  analogWrite(speedM2, abs(pwm_M2)); // set speed with pwm
  analogWrite(speedM1, abs(pwm_M1)); // set speed with pwm
}
void preciseTurn(){
  wheelBiasM1 = 0;
  wheelBiasM2 = 0;
  int error = encM2PulseNumSpeed - encM1PulseNumSpeed;
  wheelBiasM1 = abs(error);
}
static void encM1CntA_ISR() {
  encM1CntA++;
}

static void encM2CntA_ISR() {
  encM2CntA++;
}
static void echo_ISR(){
  if(ultrasonic_Edge == 0){ //Expect Rising Edge
    previousTime = micros();
    attachInterrupt(3,echoInterrupt,FALLING);
    ultrasonic_Edge = 1;
  }
  else if (ultrasonic_Edge == 1){ //Expect Falling Edge
    cm = (micros() -previousTime) /29 / 2;
    Serial.println(String(cm));
    ultrasonic_Edge = 2;
  }
}
