apt update; apt upgrade -y; apt install -y sudo git vim zsh python3 zip
wget -c https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
rm get-pip.py
apt install -y python3-dev
pip3 install -U cupy chainer chainercv
apt install -y python3-tk libglib2.0-0 libsm-dev
pip3 install -U matplotlib opencv-python
wget -c https://s3-ap-northeast-1.amazonaws.com/mycityreport/RoadDamageDataset.tar.gz
# download the caffemodel of ResNet101
wget -c "https://iuxblw.bn.files.1drv.com/y4m6GSTx36HiQ_H3VHj1-q3SlHwdCsqZGDTzoFfHFdHuMZltsJi9TtY9Ls94i9TLk5O94S5591Anhp_XZu90kSxfqJhQWcFD52Fy5PeohYFxFqfjzK_XhUItjNEPZs6WxFWHPWlKmMrpgWbdkyKfkLaZfvfPgXkvU0PbDyR_hP1aqgXaGM1zC4cdm-oHArj_kjB7bBMWYxLBJO8Yg5KrITdxw/ResNet-101-model.caffemodel?download&psid=1" -O $HOME/.chainer/dataset/pfnet/chainer/models/ResNet-101-model.caffemodel
tar xvf RoadDamageDataset.tar.gz
./merge_dasatet.sh
