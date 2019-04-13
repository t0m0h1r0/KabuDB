from kabu import StockDB

class StockDB_Analyzer(StockDB):
    def __init__(self):
        StockDB.__init__(self)
        self._rsi = self._db['RSI']
    def updateRSI(self):
        codes = [x['Code'] for x in self._codes.find()]
        self._db.begin()
        for code in codes:
            print(code)
            for data in self._prices.find(Code=code,order_by='-Date',_limit=100):
                date = data['Date']
                rsi = self.RSI(code,date)
                record = {
                'Code':code,
                'Date':date,
                'RSI':rsi[0],
                }
                self._codes.upsert(record,['Code','Date'])
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
    a.updateRSI()
