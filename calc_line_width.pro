FUNCTION calc_line_width, wavescl, spect, wingoff, fit_points=fit_points, $
    bisect_level=bisect_level, smooth_prof=smooth_prof
;-------------------------------------------------------------
;+
; NAME:
;       calc_line_width
; PURPOSE:
;       calculated the bisector positions of a line at a bisector level
;       that as determined by the fractional position between the core and
;       the wings at a selected wavelength offset from the core
; CATEGORY:
; CALLING SEQUENCE:
;       bisector_info = calc_line_width (wavelengths, profile, 0.9)
; INPUTS:
;       wavescl     = the sampling wavelengths of the spectral profile. Does not 
;                     need to be uniformly spaced.
;       spect       = measured intensities of spectral profile at given wavelength positions
;       wingoff     = wavelength offset relative to measured line core position at which t
;                     to determine reference wing intensities and subsequently line depth.
; KEYWORD PARAMETERS:
;       fit_points  = number of points around line minimum position to be used for 
;                     polynomial fitting to get line core position
;       bisect_level = fractional distance between line-core and line-wing intensities at which
;                      to make bisector measurement. Smaller values are closer to line core.
;       smooth_prof  = smoothing factor to be applied to spectral profile prior to bisector fitting. 
; OUTPUTS:
;       bisector_info =  FLTARR(8)
;                        bisector_info[0] = line core position
;                        bisector_info[1] = line core intensity
;                        bisector_info[2] = blue line wing reference intensity
;                        bisector_info[3] = red line wing reference intensity
;                        bisector_info[4] = blue wing wavelength position
;                        bisector_info[5] = red wing wavelength position
;                        bisector_info[6] = bisector position difference = line width
;                        bisector_info[7] = average bisector position   = bisector shift 
; OPTIONAL OUTPUTS:
; COMMON BLOCKS:
;
; NOTES:
;
;        Spectral line is assumed to be in absorption!
;
; MODIFICATION HISTORY:
;               2008 - KPR - original coding
;       15 Dec, 2017 - KPR - added documentation
;
;-
;-------------------------------------------------------------

IF NOT KEYWORD_SET(fit_points) THEN fit_points = 5
IF NOT KEYWORD_SET(bisect_level) THEN bisect_level = 0.5

spect_sz    = N_ELEMENTS(spect)
spect_min   = MIN(spect,spect_minpos)
;spect_minpos = spect_minpos>4<10
;spect_lc = lc_find(spect,spect_minpos - spect_sz/3,spect_minpos + spect_sz/3,11)
; determine ranges for line center fitting
cent_left   = (spect_minpos - spect_sz/4)>2
cent_right  = (spect_minpos + spect_sz/4)<(spect_sz-1)
; fit line center
spect_lc    = lc_find(spect,cent_left,cent_right,5)
; apply reasonable limits to output to correct for egregious errors
spect_lc(0) = spect_lc(0) > fit_points/2. < (spect_sz - 1 - fit_points/2.)
spect_lc(1) = spect_lc(1) > 0 < MAX(spect)

; more error checking on line center fit
; if line center position is outside of fitted range, then fit using a larger range
; it would actually probably be better if this increase was based on the value
; of the "fit_points" variable, and not fixed values
IF (spect_lc[0] GE cent_right) OR (spect_lc[0] LE cent_left) THEN BEGIN
    spect_lc = lc_find(spect,cent_left,cent_right,7)
ENDIF
; if line center position is outside of fitted range, then fit using still a larger range
IF (spect_lc[0] GE cent_right) OR (spect_lc[0] LE cent_left) THEN BEGIN
    spect_lc = lc_find(spect,cent_left,cent_right,9)
ENDIF
; if the line center position is still wacky, just use the line minimum position
IF (spect_lc[0] GE cent_right) OR (spect_lc[0] LE cent_left) THEN BEGIN
    spect_lc[0] = spect_minpos>5<(spect_sz-6)
ENDIF

; find the wavelength value corresponding to the fitted line center position
; using linear interpolation
spect_lc_wv = INTERPOL(wavescl,FINDGEN(spect_sz),spect_lc(0))

IF KEYWORD_SET(smooth_prof) THEN spect=SMOOTH(spect,smooth_prof,/EDGE)

; find the intensity values in the wings of the line profile ath the proscribed 
; distance from the fitted line core position
spect_wingint = INTERPOL(spect,wavescl,[-1,1]*wingoff + spect_lc_wv,/SPLINE)

; determine the intensity values at which to measure the bisector positions,
; for the wing intensity we take the average of the intensities in the two wings.
; first finding the depth of the line (wing minus core) and then computing 
; a fractional value of that depth (above the line minimum position).
line_depth    =  MEAN(spect_wingint) - spect_lc(1)
width_int     = (line_depth) * bisect_level + spect_lc(1)

; find the wavelength values in each wing at the bisector intensity value determined above
; this uses linear interpolation - quadratic or spline interpolation might
; give a slight improvement, but in the steep wings of most lines (especially H-alpha)
; linear interpolation is probably accurate enough.
blue_wing = INTERPOL(wavescl(0:spect_lc(0)),spect(0:spect_lc(0)),width_int)
red_wing = INTERPOL(wavescl(spect_lc(0):*),spect(spect_lc(0):*),width_int)

RETURN,[spect_lc(0),spect_lc(1),spect_wingint[0],spect_wingint[1], blue_wing, red_wing, $
        red_wing - blue_wing, (red_wing + blue_wing)/2.]

END
