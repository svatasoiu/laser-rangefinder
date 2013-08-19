boolean on = false;
int laserPin = 31;
int buttonPin = 33;

void setup() {
  pinMode(laserPin,OUTPUT);
  pinMode(buttonPin,INPUT);
}

void loop() {
  if (digitalRead(buttonPin) == 1) {
    on = !on;
    digitalWrite(laserPin, on);
    while (digitalRead(buttonPin) == 1) {
      continue;
    }
  }
}
