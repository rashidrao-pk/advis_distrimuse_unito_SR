## Check Status of Services Running

```bash
sudo systemctl status inference.service
```

- Stop it now:

```bash
sudo systemctl stop inference.service
```

- Start again:

```bash
sudo systemctl start inference.service
```

- Restart:

```bash
sudo systemctl restart inference.service
```

- Disable automatic startup at boot:

```bash
sudo systemctl disable inference.service
```

- Enable automatic startup at boot:

```bash
sudo systemctl enable inference.service
```

- If you want to stop it now and also prevent it from starting automatically after reboot, use:

```bash
sudo systemctl disable --now inference.service
```

- To re-enable and immediately start it:

```bash
sudo systemctl enable --now inference.service
```

- verify whether auto-start is currently enabled:

```bash
systemctl is-enabled inference.service
systemctl is-active inference.service
```

- For logs:

```bash
sudo journalctl -u inference.service -f
```

- verify `vlan_setup.service`

```bash
systemctl status vlan_setup.service inference.service
```

## RUN AGAIN AFTER LIVE TEST

```bash
sudo systemctl enable --now inference.service
```
