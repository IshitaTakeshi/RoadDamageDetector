apt update; apt upgrade -y; apt install -y sudo git vim zsh python3 zip
wget -c https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
apt install -y python3-dev
pip3 install cupy chainer chainercv
apt install -y python3-tk libglib2.0-0 libsm-dev
pip3 install matplotlib opencv-python
wget -c https://s3-ap-northeast-1.amazonaws.com/mycityreport/RoadDamageDataset.tar.gz
tar xvf RoadDamageDataset.tar.gz
./merge_dasatet.sh
