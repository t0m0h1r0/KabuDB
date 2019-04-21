from kabu import StockDB
import itertools as it

class StockDB_Analyzer(StockDB):
    def __init__(self):
        StockDB.__init__(self)
        self._EMA_coff = self._calc_EMA_coff()
        self._funcs = {
        'RSI':self.RSI,
        'EMA':self.EMA,
        'MACD':self.MACD,
        'AVE':self.Average,
        }

    def _calc_EMA_coff(self):
        ret = {}
        for n in [5,20,40]:
            alpha = 2./(float(n)+1)
            count = it.count()
            coff = []
            for x in count:
                c = alpha*(pow(1-alpha,x))
                coff.append(c)
                if sum(coff)>0.999:
                    ret[n]=coff
                    break
        return ret


    def update(self):
        codes = [x['Code'] for x in self._codes.find()]

        self._db.begin()
        for code in codes:
            for name,func in self._funcs.items():
                f_dates = set(x['Date'] for x in self._prices.find(Code=code,order_by='-Date',_limit=200))
                t_dates = set(x['Date'] for x in self._db[name].find(Code=code,order_by='-Date',_limit=200))
                for date in f_dates - t_dates:
                    record = {
                    'Code':code,
                    'Date':date,
                    }
                    ans = func(code,date)
                    record2 = {
                    '{}_{}'.format(name,n):an for n,an in enumerate(ans)
                    }
                    record.update(record2)
                    self._db[name].upsert(record,['Code','Date'])
        self._db.commit()

    def Average(self,code,date):
        ret = []
        for term in [5,25,75]:
            result = self._prices.find(
            Code=code,
            Date={'<=':date},
            order_by='-Date',
            _limit=term,
            )
            data = [x['End'] for x in result]
            ret.append(float(sum(data))/float(len(data)))
        return ret

    def MACD(self,code,date):
        data = self.EMA(code,date)
        return data[0]-data[1],data[1]-data[2]

    def EMA(self,code,date):
        ret = []
        for days,ema in sorted(self._EMA_coff.items()):
            result = self._prices.find(
            Code=code,
            Date={'<=':date},
            order_by='-Date',
            _limit=len(ema),
            )
            data = [x['End'] for x in result]
            res = sum(map(lambda x:x[0]*x[1],zip(ema,data)))/sum(ema)
            ret.append(res)
        return ret

    def RSI(self,code,date):
        ret = []
        for term in [14]:
            result = self._prices.find(
            Code=code,
            Date={'<=':date},
            order_by='-Date',
            _limit=term+1,
            )
            data = [x['End'] for x in result]
            diff = list(map(lambda x:x[0]-x[1],zip(data,data[1:])))
            a = list(filter(lambda x:x>0.,diff))
            b = list(filter(lambda x:x<0.,diff))
            try:
                rsi = 100.*sum(a)/(sum(a)-sum(b))
            except:
                rsi = 50.0
                print(code,date,data,diff)
            ret.append(rsi)

        return ret

if __name__ == '__main__':
    import datetime
    today = int('{0:%Y%m%d}'.format(datetime.date.today()))

    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-r','--rsi',action='store_true')
    parser.add_argument('-a','--average',action='store_true')
    parser.add_argument('-d','--date',type=int,default=today)
    parser.add_argument('-c','--codes',nargs='*',type=int,default=[6501,])
    args = parser.parse_args()
    print(args)

    a=StockDB_Analyzer()
    a.update()
