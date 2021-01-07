# RaspiCar_treelet
Put CarVerification RL models trained in Unity simulation environment on a real raspicar
## Usage
* GRPC is used to communicate between computer and raspicar. As the client side, raspicar sends the current image to computer for action. As the server side, the computer receives the image input to the pre-trained RL model to choose action and return it to raspicar
* Raspicar system runs run.py
* Computer runs server.py
