# install basic tools (considering running on docker)
apt update; apt upgrade -y
apt install -y sudo git vim zsh python3 zip wget

# install pip
wget -c https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
rm get-pip.py

# install chainercv and other tools
apt install -y python3-dev
pip3 install -U chainercv==v0.9.0

apt install -y python3-tk libglib2.0-0 libsm-dev
pip3 install -U matplotlib opencv-python

# download and setup dataset
wget -c https://s3-ap-northeast-1.amazonaws.com/mycityreport/RoadDamageDataset.tar.gz
tar xvf RoadDamageDataset.tar.gz
./merge_dasatet.sh
