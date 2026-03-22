## Notes 3/7/26

Procedural notes as of now
	- Board is powered through computer, will test with external power later

Currently using STM32F407G-DISC1 **discovery board**
	- Discovery board includes digital microphone which will not be used.
	- Discovery board includes audio DAC with integrated class D speaker driver. Consider using this circuit for custom STM board?
	- Also has audio mini-jack converter directly from DAC, which is currently used for testing purposes.

# More on Discovery board DAC use (from manual):

The STM32F407VG microcontroller controls the audio DAC through the I2C interface and processes digital signals through an I2S connection or an analog input signal.

• The sound can come independently from different inputs:

	– MEMS microphone: digital using PDM protocol or analog when using the low pass filter
	– USB connector: from external mass storage such as a USB key, USB HDD and others
	– Internal memory of the STM32F407VG microcontroller

• The sound can be output in different ways through the audio DAC:

	– Using I2S protocol
	– Using DAC to analog input IN1x of the audio DAC
	– Using the microphone output directly via a low-pass filter to analog input IN2x of the audio DAC

# Current mic modules being used: SPW 2430 Si Mic
https://www.adafruit.com/product/2716#technical-details

The SPW2430 is a small, low cost MEMS mic with a range of **100Hz - 10KHz**
	- Will need to adjust various parameters for actual mic with up to 20kHz (human hearing) range

Usage:
	- the output peak-to-peak voltage has a 0.67V DC bias and about 100mVpp (peak-to-peak) when talking near the microphone, which is good for attaching to something that expects 'line level' input without clipping. The peak-to-peak can be as high as 1Vpp if there's a very loud sound.
	- connect GND to ground, Vin to 3.3-5VDC. For the best performance, use the "quietest" supply available (on an Arduino, this would be the 3.3V supply). The audio waveform will come out of the DC pin. The output will have a DC bias of 0.67V so when its perfectly quiet that's what you'll read, there's a little drift. If the audio equipment you're using requires AC coupled audio, you can grab the signal out of the AC pin, which has a 10uF capacitor in series.
	-  If you're connecting to a microcontroller pin, you don't need an amplifier or decoupling capacitor - connect the DC pin directly to the microcontroller ADC pin.
