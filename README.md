# Drowzee
![Image](resources/Drowzee.png)

Drowsiness detection system. For running create a virtualenv with the [requirements](requirements.txt) and python 3.6.8 (this is the version tested). This are the commands for Mac.

```
python3 -m venv name
source name/bin/activate
pip install -r requirements.txt
```

For running the app just need to execute this commands:
```
PYTHONPATH=. python scripts/app.py -th 0.2 -frame 12
```
For quiting just press the key "q".

For this experiment I used has ey treshold 0.2 and 12 frames with closed eyes for the alarm to pop.

![Image](resources/drowzee_example.gif)
