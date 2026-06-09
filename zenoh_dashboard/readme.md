## Receive messages from ADVIS for Non-Safety related messages 

- Use of Zenoh
- Publish two messages:
    - M1: Anomaly Maps
    - M2: Detection Scores to plot Timeline


> Using TCP from SR --> UniTo --> SR 


Run Message Receiving

### Receive M1:
```bash
python dashboard_viewer.py
```


### Receive M2:
```bash
python timeline_viewer.py
```


