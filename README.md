# peopleTracker
This repo has been tested with cuda11.3 and pytorch 1.12.0 with python 3.8.12 in a seperate conda env.\\

Steps:
1. `git clone git@github.com:niveditarufus/people_tracker.git`
2. `cd people tracker`
3. `git clone git@github.com:niveditarufus/ultralytics.git`
4. `conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch`
5. copy [model.zip](https://drive.google.com/file/d/1lqU2knL9B2NbaIkKukE1liH-J7avo3Gh/view?usp=share_link) to root folder
6. copy [reid_model](https://drive.google.com/file/d/1rsl29uKXCr7YxJRGZAiEpoqjhDRXC440/view?usp=share_link) `people_tracker/reid/`
7. run `python3 inference.py` with path to your data
