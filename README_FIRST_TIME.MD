

```bash
# Connect Server
ssh -Y unito@distrimuse

```



### 1.1 Retreive Codes 

```bash

cd ~/advis/
git clone https://github.com/rashidrao-pk/advis_distrimuse_unito_SR
cd ~/advis/advis_distrimuse_unito_SR
```







### 1.2 Tranport Data

Ship Datasets to server

**Dataset v2**

```bash

# Copy Raw Videos 

scp D:\DS\SR\v2\camera1_20251210_142154_normal.mp4 unito@distrimuse:/home/unito/advis/DS/SR/v2

scp D:\DS\SR\v2\camera1_20251210_150045_box_fall.mp4 unito@distrimuse:/home/unito/advis/DS/SR/v2

scp D:\DS\SR\v2\camera1_20251210_151444_fallen_operator.mp4 unito@distrimuse:/home/unito/advis/DS/SR/v2

## Copy Processed Data
scp D:\DS\SR\v2\processed.zip unito@distrimuse:/home/unito/advis/DS/SR/v2
scp D:\DS\SR\v2\refined.zip unito@distrimuse:/home/unito/advis/DS/SR/v2
scp D:\DS\SR\v2\masks.zip unito@distrimuse:/home/unito/advis/DS/SR/v2

### TEST Data
scp D:\DS\SR\v2\camera1_20251210_150045_box_fall.zip unito@distrimuse:/home/unito/advis/DS/SR/v2
scp D:\DS\SR\v2\camera1_20251210_151444_fallen_operator.zip unito@distrimuse:/home/unito/advis/DS/SR/v2


cd /home/unito/advis/DS/SR/v2
unzip processed.zip -d /home/unito/advis/DS/SR/v2
unzip refined.zip -d /home/unito/advis/DS/SR/v2
unzip masks.zip -d /home/unito/advis/DS/SR/v2

unzip camera1_20251210_150045_box_fall.zip -d /home/unito/advis/DS/SR/v2
unzip camera1_20251210_151444_fallen_operator.zip -d /home/unito/advis/DS/SR/v2

rm camera1_20251210_150045_box_fall.zip
rm camera1_20251210_151444_fallen_operator.zip

```

#### ONLY ONCE
```bash
# INTALL DISPLAY Settings 
sudo apt install -y libxcb-cursor0
pixi run ros2 topic list
pixi run ros2 topic hz /camera/front_view/image_raw
pixi run ros2 topic echo /camera/front_view/image_raw --once
```







**Dataset V3**

```bash
scp D:/DS/SR/v3/recording_20260313_133316.zip unito@HPZ2miniDistriMuSe:
cd /home/unito/distrimuse
mkdir -p bags
unzip recording_20260313_133316.zip -d bags
find /home/unito/advis/bags/recording_20260313_133316 -maxdepth 2 -type f
```



**Check if OPen CV works**

```bash
# or

## check if the display for images is working
pixi run python -c "import cv2, numpy as np; img=np.zeros((300,300,3)); cv2.imshow('test', img); cv2.waitKey(0)"
```

```bash
cp ~/advis/advis_distrimuse_unito_SR/scripts/pixi_save_frames.py ~/advis/distrimuse-image-broadcaster
```



follow script or create to save_frames
```bash
nano pixi_save_frames.py
```



1️⃣  Custom save directory + one camera

```bash
pixi run python pixi_save_frames.py --ros-args -p save_dir:=/home/unito/advis/back_frames -p topics:="['/camera/back_view/image_raw']"
```

3️⃣ Multiple cameras 🔥

```bash
pixi run python pixi_save_frames.py --ros-args -p save_dir:=/home/unito/advis/all_frames -p topics:="['/camera/front_view/image_raw','/camera/back_view/image_raw','/camera/side_view/image_raw']"

```




```bash
# Verify saved frames
ls /home/unito/advis/back_frames | head 
```