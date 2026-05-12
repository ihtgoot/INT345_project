# how is distortion generated
* as hot air have different refractive index and the layrin of such regins can couse light to bend thus causing distortion
* scene point → random shifted location + smeared energy
* x′=x+δ(x) , δ(x) is displacement factor
* in turbulence , Blur is NOT the same everywhere. So output pixel = mix of nearby pixels  BUT the way you mix them depends on location
* I_blur(x) = sum of nearby pixels × weights | weights = different for each pixel x
* output pixel value = weighted sum of nearby input pixels => The intensity at a pixel is computed by taking nearby pixel values and combining them using weights that describe how much the original light has spread.
* 