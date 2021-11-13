import pandas as pd
import holidays
import math

class bulid_wfe:
    def __init__(self,timestamp):
        self.timestamp=timestamp


    def wk_fuzzy(self,wk):
        us_holidays = holidays.US(state='NY')

        if self.timestamp not in us_holidays:
            if wk in ['0', '6']:
                ek_emb = [math.cos(-math.pi / 3), math.sin(-math.pi / 3)]
            else:
                ek_emb = [math.cos(math.pi / 2), math.sin(math.pi / 2)]
        else:
            ek_emb = [math.cos(-math.pi*2 / 3), math.sin(-math.pi*2 / 3)]

        return ek_emb


    def hr_fuzzy(self,hr):
        if hr>=0.5 and hr<1.5:
            e_hr_from = [math.cos(math.pi / 2),math.sin(math.pi / 2)]
            e_hr_to = [math.cos(math.pi / 3), math.sin(math.pi / 3)]
            memship_from = 1.5-hr
            memship_to = 1-memship_from
            hr_fuzzy = [e_hr_from[0]*memship_from+e_hr_to[0]*memship_to, e_hr_from[1]*memship_from+e_hr_to[1]*memship_to]
            return hr_fuzzy
        elif hr>=1.5 and hr<2.5:
            e_hr_to = [math.cos(math.pi / 3), math.sin(math.pi / 3)]
            return e_hr_to
        elif hr>=2.5 and hr<3.5:
            e_hr_from = [math.cos(math.pi / 3), math.sin(math.pi / 3)]
            e_hr_to = [math.cos(math.pi / 6), math.sin(math.pi / 6)]
            memship_from = 3.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0]*memship_from+e_hr_to[0]*memship_to, e_hr_from[1]*memship_from+e_hr_to[1]*memship_to]
            return hr_fuzzy
        elif hr>=3.5 and hr<4.5:
            e_hr_to = [math.cos(math.pi / 6), math.sin(math.pi / 6)]
            return e_hr_to
        elif hr>=4.5 and hr<5.5:
            e_hr_from = [math.cos(math.pi / 6), math.sin(math.pi / 6)]
            e_hr_to = [math.cos(0), math.sin(0)]
            memship_from = 5.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0]*memship_from+e_hr_to[0]*memship_to, e_hr_from[1]*memship_from+e_hr_to[1]*memship_to]
            return hr_fuzzy
        elif hr >= 5.5 and hr < 6.5:
            e_hr_to = [math.cos(0), math.sin(0)]
            return e_hr_to
        elif hr >= 6.5 and hr < 7.5:
            e_hr_from = [math.cos(0), math.sin(0)]
            e_hr_to = [math.cos(-math.pi / 6), math.sin(-math.pi / 6)]
            memship_from = 7.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0]*memship_from+e_hr_to[0]*memship_to, e_hr_from[1]*memship_from+e_hr_to[1]*memship_to]
            return hr_fuzzy
        elif hr >= 7.5 and hr < 8.5:
            e_hr_to = [math.cos(-math.pi / 6), math.sin(-math.pi / 6)]
            return e_hr_to
        elif hr >= 8.5 and hr < 9.5:
            e_hr_from = [math.cos(-math.pi / 6), math.sin(-math.pi / 6)]
            e_hr_to = [math.cos(-math.pi / 3), math.sin(-math.pi / 3)]
            memship_from = 9.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0]*memship_from+e_hr_to[0]*memship_to, e_hr_from[1]*memship_from+e_hr_to[1]*memship_to]
            return hr_fuzzy
        elif hr >= 9.5 and hr < 10.5:
            e_hr_to = [math.cos(-math.pi / 3), math.sin(-math.pi / 3)]
            return e_hr_to
        elif hr >= 10.5 and hr < 11.5:
            e_hr_from = [math.cos(-math.pi / 3), math.sin(-math.pi / 3)]
            e_hr_to = [math.cos(-math.pi / 2), math.sin(-math.pi / 2)]
            memship_from = 11.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0]*memship_from+e_hr_to[0]*memship_to, e_hr_from[1]*memship_from+e_hr_to[1]*memship_to]
            return hr_fuzzy
        elif hr >= 11.5 and hr < 12.5:
            e_hr_to = [math.cos(-math.pi / 2), math.sin(-math.pi / 2)]
            return e_hr_to
        elif hr >= 12.5 and hr < 13.5:
            e_hr_from = [math.cos(-math.pi / 2), math.sin(-math.pi / 2)]
            e_hr_to = [math.cos(-math.pi*2 / 3), math.sin(-math.pi*2 / 3)]
            memship_from = 13.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0]*memship_from+e_hr_to[0]*memship_to, e_hr_from[1]*memship_from+e_hr_to[1]*memship_to]
            return hr_fuzzy
        elif hr >= 13.5 and hr < 14.5:
            e_hr_to = [math.cos(-math.pi * 2 / 3), math.sin(-math.pi * 2 / 3)]
            return e_hr_to
        elif hr >= 14.5 and hr < 15.5:
            e_hr_from = [math.cos(-math.pi * 4 / 6), math.sin(-math.pi * 4 / 6)]
            e_hr_to = [math.cos(-math.pi * 5 / 6), math.sin(-math.pi * 5 / 6)]
            memship_from = 15.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0] * memship_from + e_hr_to[0] * memship_to, e_hr_from[1] * memship_from + e_hr_to[1] * memship_to]
            return hr_fuzzy
        elif hr >= 15.5 and hr < 16.5:
            e_hr_to = [math.cos(-math.pi * 5 / 6), math.sin(-math.pi * 5 / 6)]
            return e_hr_to
        elif hr >= 16.5 and hr < 17.5:
            e_hr_from = [math.cos(-math.pi * 5 / 6), math.sin(-math.pi * 5 / 6)]
            e_hr_to = [math.cos(-math.pi * 6 / 6), math.sin(-math.pi * 6 / 6)]
            memship_from = 17.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0]*memship_from+e_hr_to[0]*memship_to, e_hr_from[1]*memship_from+e_hr_to[1]*memship_to]
            return hr_fuzzy
        elif hr >= 17.5 and hr < 18.5:
            e_hr_to = [math.cos(-math.pi * 6 / 6), math.sin(-math.pi * 6 / 6)]
            return e_hr_to
        elif hr >= 18.5 and hr < 19.5:
            e_hr_from = [math.cos(-math.pi * 6 / 6), math.sin(-math.pi * 6 / 6)]
            e_hr_to = [math.cos(math.pi * 5 / 6), math.sin(math.pi * 5 / 6)]
            memship_from = 19.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0] * memship_from + e_hr_to[0] * memship_to, e_hr_from[1] * memship_from + e_hr_to[1] * memship_to]
            return hr_fuzzy
        elif hr >= 19.5 and hr < 20.5:
            e_hr_to = [math.cos(math.pi * 5 / 6), math.sin(math.pi * 5 / 6)]
            return e_hr_to
        elif hr >= 20.5 and hr < 21.5:
            e_hr_from = [math.cos(math.pi * 5 / 6), math.sin(math.pi * 5 / 6)]
            e_hr_to = [math.cos(math.pi * 4 / 6), math.sin(math.pi * 4 / 6)]
            memship_from = 21.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0] * memship_from + e_hr_to[0] * memship_to, e_hr_from[1] * memship_from + e_hr_to[1] * memship_to]
            return hr_fuzzy
        elif hr >= 21.5 and hr < 22.5:
            e_hr_to = [math.cos(math.pi * 4 / 6), math.sin(math.pi * 4 / 6)]
            return e_hr_to
        elif hr >= 22.5 and hr < 23.5:
            e_hr_from = [math.cos(math.pi * 4 / 6), math.sin(math.pi * 4 / 6)]
            e_hr_to = [math.cos(math.pi * 3 / 6), math.sin(math.pi * 3 / 6)]
            memship_from = 23.5 - hr
            memship_to = 1 - memship_from
            hr_fuzzy = [e_hr_from[0] * memship_from + e_hr_to[0] * memship_to, e_hr_from[1] * memship_from + e_hr_to[1] * memship_to]
            return hr_fuzzy
        elif hr >= 23.5 and hr <= 24 or hr>=0 and hr<0.5:
            e_hr_to = [math.cos(math.pi * 3 / 6), math.sin(math.pi * 3 / 6)]
            return e_hr_to


    def cfeVec(self):
        hr = int(pd.to_datetime(self.timestamp).strftime("%H"))
        mt = int(pd.to_datetime(self.timestamp).strftime("%M"))
        mtdec = mt/60
        hrdec = hr+mtdec

        wk = pd.to_datetime(self.timestamp).strftime("%w")

        hr_fuzzy = self.hr_fuzzy(hrdec)
        wk_fuzzy = self.wk_fuzzy(wk)

        # Normalization from [-1,1] to [0,1]
        hr_fuzzy_ = [(hr_fuzzy[0]+1)*0.5, (hr_fuzzy[1]+1)*0.5]
        wk_fuzzy_ = [(wk_fuzzy[0] + 1) * 0.5, (wk_fuzzy[1] + 1) * 0.5]

        return wk_fuzzy_[0],wk_fuzzy_[1],hr_fuzzy_[0],hr_fuzzy_[1]
