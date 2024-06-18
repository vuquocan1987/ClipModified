python -m venv universal
source universal/bin/activate
pip3 install -r requirements.txt
cd pretrained_weights/
apt install wget
wget https://www.dropbox.com/s/jdsodw2vemsy8sz/swinunetr.pth