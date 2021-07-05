## I. Set up environment
Run these command
```
python3 -m venv env
source env/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```
## II. Run demo
Open new terminal, run server by command:
```
python -m app.run_server
```
Wait until terminal print "Server waiting at:  56788"<br>
Open new terminal, run inference by command:
```
python -m app.infer --text [text to predict] --html [path to wiki page]
```
## III. Training
Config dataset dir in **config.yaml**<br>
Run training using command:
```
python -m training.train
```
## IV. Evaluation
Export h5 file to pb file: change h5 path in file **h5_to_pb.py** on line 6, run:
```
python h5_to_pb.py
```
Run evaluation using command:
```
python -m training.eval --pretrained [path_to_pb]
```
## III. Run demo
Pull pretrained: git lfs pull<br>
Change pb path in config.yaml, line 22<br>
CHnage file labels.txt path in config.yaml, line 23<br>
Open new terminal, run server by command:
```
python -m app.run_server
```
Open new terminal, run inference by command:
```
python -m app.infer --text [text to predict] --html [path to wiki page]



