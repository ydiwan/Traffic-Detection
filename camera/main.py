import sensor, image, time, ustruct
from pyb import USB_VCP

# Initialize the USB Virtual COM Port
usb = USB_VCP()

# Set up the camera sensor
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.VGA) # 320x240 resolution keeps the USB transfer fast
sensor.skip_frames(time = 2000)

while(True):
    # Wait for the PC to send the 'snap' command
    cmd = usb.recv(4, timeout=10)
    
    if (cmd == b'snap'):
        # Take a picture and compress it into a JPEG
        img = sensor.snapshot().compress(quality=80)
        
        # Send the 4-byte size of the image first, then the actual image data
        usb.send(ustruct.pack("<L", img.size()))
        usb.send(img)