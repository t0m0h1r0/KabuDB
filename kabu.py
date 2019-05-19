import dataset

class StockDB:
    def __init__(self,filename='kabu.db'):
        self._filename=filename
        self._db = dataset.connect('sqlite:///{0}'.format(filename))
        self._codes = self._db['Codes']
        self._prices = self._db['Prices']
        self._keys = ['Code','Date','Begin','High','Low','End','Amount']
        self._jpxlist = {}
    def getClose(self,code=6501,date=20190401,term=15):
        result = self._prices.find(
        Code=code,
        Date={'<=':date},
        order_by='-Date',
        _limit=term,
        )
        data = [x['End'] for x in result]
        return data
    def getOHLC(self,code=6501,date=20190401,term=15):
        result = self._prices.find(
        Code=code,
        Date={'<=':date},
        order_by='-Date',
        _limit=term,
        )
        data = [(x['Date'],x['Begin'],x['High'],x['Low'],x['End']) for x in result]
        return data


if __name__ == '__main__':
    pass
