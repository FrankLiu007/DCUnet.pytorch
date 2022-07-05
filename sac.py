import numpy as np

sac_head = np.dtype([
    ('delta', 'f4'),  # RF time increment, sec    #
    ('depmin', 'f4'),  # minimum amplitude      #
    ('depmax', 'f4'),  # maximum amplitude      #
    ('scale', 'f4'),  # amplitude scale factor #
    ('odelta', 'f4'),  # observed time inc      #
    ('b', 'f4'),
    ('e', 'f4'),
    ('o', 'f4'),
    ('a', 'f4'),
    ('internal1', 'f4'),
    ('t0', 'f4'),  # user-defined time pick #
    ('t1', 'f4'),  # user-defined time pick #
    ('t2', 'f4'),  # user-defined time pick #
    ('t3', 'f4'),  # user-defined time pick #
    ('t4', 'f4'),  # user-defined time pick #
    ('t5', 'f4'),  # user-defined time pick #
    ('t6', 'f4'),  # user-defined time pick #
    ('t7', 'f4'),  # user-defined time pick #
    ('t8', 'f4'),  # user-defined time pick #
    ('t9', 'f4'),  # user-defined time pick #
    ('f', 'f4'),  # user-defined time pick #
    ('resp0', 'f4'),  # instrument respnse parm#
    ('resp1', 'f4'),  # instrument respnse parm#
    ('resp2', 'f4'),  # instrument respnse parm#
    ('resp3', 'f4'),  # instrument respnse parm#
    ('resp4', 'f4'),  # instrument respnse parm#
    ('resp5', 'f4'),  # instrument respnse parm#
    ('resp6', 'f4'),  # instrument respnse parm#
    ('resp7', 'f4'),  # instrument respnse parm#
    ('resp8', 'f4'),  # instrument respnse parm#
    ('resp9', 'f4'),  # instrument respnse parm#
    ('stla', 'f4'),  # T station latititude     #
    ('stlo', 'f4'),  # T station longitude      #
    ('stel', 'f4'),  # T station elevation, m   #
    ('stdp', 'f4'),  # T station depth, m      #
    ('evla', 'f4'),  # event latitude         #
    ('evlo', 'f4'),  # event longitude         #
    ('evel', 'f4'),  # event elevation         #
    ('evdp', 'f4'),  # event depth         #
    ('mag', 'f4'),  # reserved for future use#
    ('user0', 'f4'),  # available to user      #
    ('user1', 'f4'),  # available to user      #
    ('user2', 'f4'),  # available to user      #
    ('user3', 'f4'),  # available to user      #
    ('user4', 'f4'),  # available to user      #
    ('user5', 'f4'),  # available to user      #
    ('user6', 'f4'),  # available to user      #
    ('user7', 'f4'),  # available to user      #
    ('user8', 'f4'),  # available to user      #
    ('user9', 'f4'),  # available to user      #
    ('dist', 'f4'),  # stn-event distance, km #
    ('az', 'f4'),  # event-stn azimuth      #
    ('baz', 'f4'),  # stn-event azimuth      #
    ('gcarc', 'f4'),  # stn-event dist, degrees#
    ('internal2', 'f4'),  # internal use           #
    ('internal3', 'f4'),  # internal use           #
    ('depmen', 'f4'),  # internal use           #
    ('cmpaz', 'f4'),  # event-stn azimuth      #
    ('cmpinc', 'f4'),  # event-stn azimuth      #
    ('xminimum', 'f4'),  # event-stn azimuth      #
    ('xmaximum', 'f4'),  # event-stn azimuth      #
    ('yminimum', 'f4'),  # event-stn azimuth      #
    ('ymaximum', 'f4'),  # event-stn azimuth      #
    ('unused6', 'f4'),  # reserved for future use#
    ('unused7', 'f4'),  # reserved for future use#
    ('unused8', 'f4'),  # reserved for future use#
    ('unused9', 'f4'),  # reserved for future use#
    ('unused10', 'f4'),  # reserved for future use#
    ('unused11', 'f4'),  # reserved for future use#
    ('unused12', 'f4'),  # reserved for future use#
    ("nzyear", "i4"),  # F zero time of file, yr  #
    ("nzjday", "i4"),  # F zero time of file, day #
    ("nzhour", "i4"),  # F zero time of file, hr  #
    ("nzmin", "i4"),  # F zero time of file, min #
    ("nzsec", "i4"),  # F zero time of file, sec #
    ("nzmsec", "i4"),  # F zero time of file, msec#
    ("nvhdr", "i4"),  # R header version number  #
    ("norid", "i4"),  # Origin ID (CSS 3.0)        #
    ("nevid", "i4"),  # Event ID (CSS 3.0)	         #
    ("npts", "i4"),  # RF number of samples      #
    ("internal7", "i4"),  # internal use           #
    ("nwfid", "i4"),  # Waveform ID (CSS 3.0)           #
    ("nxsize", "i4"),  # Spectral Length (Spectral files only)#
    ("nysize", "i4"),  # Spectral Width (Spectral files only)#
    ("unused15", "i4"),  # reserved for future use#
    ("iftype", "i4"),  # RA type of file          #
    ("idep", "i4"),  # type of amplitude      #
    ("iztype", "i4"),  # zero time equivalence  #
    ("unused16", "i4"),  # reserved for future use#
    ("iinst", "i4"),  # recording instrument   #
    ("istreg", "i4"),  # stn geographic region  #
    ("ievreg", "i4"),  # event geographic region#
    ("ievtyp", "i4"),  # event type             #
    ("iqual", "i4"),  # quality of data        #
    ("isynth", "i4"),  # synthetic data flag    	#
    ("imagtyp", "i4"),  # Magnitude type:			#
    ("imagsrc", "i4"),  # Source of magnitude information:#
    ("unused19", "i4"),  # reserved for future use#
    ("unused20", "i4"),  # reserved for future use#
    ("unused21", "i4"),  # reserved for future use#
    ("unused22", "i4"),  # reserved for future use#
    ("unused23", "i4"),  # reserved for future use#
    ("unused24", "i4"),  # reserved for future use#
    ("unused25", "i4"),  # reserved for future use#
    ("unused26", "i4"),  # reserved for future use#
    ("leven", "i4"),  # RA data-evenly-spaced flag#
    ("lpspol", "i4"),  # station polarity flag  #
    ("lovrok", "i4"),  # overwrite permission   #
    ("lcalda", "i4"),  # calc distance, azimuth #
    ("unused27", "i4"),  # reserved for future use#
    ("kstnm", "S8"),  # F station name           #
    ("kevnm", "S16"),  # event name             #
    ("kevnm1", "S8"),  # event name             #
    ("khole", "S8"),  # man-made event name    #
    ("ko", "S8"),  # event origin time id   #
    ("ka", "S8"),  # 1st arrival time ident #
    ("kt0", "S8"),  # time pick 0 ident      #
    ("kt1", "S8"),  # time pick 1 ident      #
    ("kt2", "S8"),  # time pick 2 ident      #
    ("kt3", "S8"),  # time pick 3 ident      #
    ("kt4", "S8"),  # time pick 4 ident      #
    ("kt5", "S8"),  # time pick 5 ident      #
    ("kt6", "S8"),  # time pick 6 ident      #
    ("kt7", "S8"),  # time pick 7 ident      #
    ("kt8", "S8"),  # time pick 8 ident      #
    ("kt9", "S8"),  # time pick 9 ident      #
    ("kf", "S8"),  # end of event ident     #
    ("kuser0", "S8"),  # available to user      #
    ("kuser1", "S8"),  # available to user      #
    ("kuser2", "S8"),  # available to user      #
    ("kcmpnm", "S8"),  # F component name         #
    ("knetwk", "S8"),  # network name           #
    ("kdatrd", "S8"),  # date data read         #
    ("kinst", "S8")  # instrument name        #
]
)


def read_sac(path):
    f = open(path, "rb")
    head = np.fromfile(f, dtype=sac_head, count=1)[0]
    da = np.fromfile(f, dtype=np.float32, count=head["npts"])

    f.close()

    return head, da


####---------------- write sac, --------------

def write_sac(head, data, path):
    f = open(path, "wb")
    head.tofile(f)
    data.tofile(f)
    f.close()
