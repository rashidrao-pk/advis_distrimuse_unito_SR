## Upload Rosbags

## Upload processed Test data

```bash
zip /Users/rashid/data/DS/SR/v6/Jul27/test

scp /Users/rashid/data/DS/SR/v6/Jul27/test.zip mrashid@slurm.hpc4ai.unito.it:/beegfs/home/mrashid/datasets/AD/SR/V6

# Upload test data

unzip -o /beegfs/home/mrashid/datasets/AD/SR/V6/test.zip \
  -d /beegfs/home/mrashid/datasets/AD/SR/V6/test \
  -x "__MACOSX/*"

ls -lah /beegfs/home/mrashid/datasets/AD/SR/V6/test
du -sh /beegfs/home/mrashid/datasets/AD/SR/V6/test

```
