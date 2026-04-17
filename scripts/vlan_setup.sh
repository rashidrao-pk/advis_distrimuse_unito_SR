#!/bin/bash

cd /home/unito/dm/distrimuse-seds/
source setup_ros.sh vlans.conf unito/dm kilted
./vlan_manager.sh vlans.conf