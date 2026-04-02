import psutil
import platform

# CPU Info
print("Processor:", platform.processor())
print("Physical cores:", psutil.cpu_count(logical=False))
print("Total cores (logical):", psutil.cpu_count(logical=True))
print("CPU Frequency:", psutil.cpu_freq())

# RAM Info
ram = psutil.virtual_memory()
print("Total RAM (GB):", round(ram.total / (1024**3), 2))
print("Available RAM (GB):", round(ram.available / (1024**3), 2))
print("Used RAM (GB):", round(ram.used / (1024**3), 2))
