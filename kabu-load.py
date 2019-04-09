import urllib3
urllib3.disable_warnings()
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor as PoolExec

import datetime as dt
import re
import dataset
import itertools as it
import io
import pandas,tabula,numpy

from functools import partial

class StockDB:
    def __init__(self,filename='kabu.db'):
        self._filename=filename
        self._db = dataset.connect('sqlite:///{0}'.format(filename))
        self._codes = self._db['Codes']
        self._prices = self._db['Prices']
        self._keys = ['Code','Date','Begin','High','Low','End','Amount']
        self._jpxlist = {}

    def stock_cvs_parse(self,line):
        index = [
        ('Year',int),('Month',int),('Day',int),
        ('Begin',float),('High',float),('Low',float),('End',float),
        ('Amount',int),
        ('Adjust',float),
        ]
        x = list(filter(lambda x:x!='', re.split('[-,\"\n]',line)))
        y=map(lambda x:(x[1],x[2](x[0])),zip(x,*zip(*index)))
        z=dict(zip(*zip(*y)))
        z.update({'Date':z['Year']*10000+z['Month']*100+z['Day']})
        return z

    def downloadStock(self,year=2018,code=6501):
        http = urllib3.PoolManager(32)
        url = 'https://kabuoji3.com/stock/file.php'
        data = {'code':str(code),'year':str(year)}
        r = http.request('POST',url,fields=data,timeout=10.0,retries=10)
        if r.status == 404:
            raise
        csv = r.data.decode('shift-jis')
        f=io.StringIO(csv)
        #print(r.data.decode('shift-jis'))
        ret=[]
        for n,line in enumerate(f):
            if n<2:
                continue
            x = self.stock_cvs_parse(line)
            x.update({'Code':int(code)})
            ret.append(x)
        return ret

    def setStock(self,data):
        _data = []
        for x in data:
            elem = {}
            for key in self._keys:
                if not key in x:
                    raise
                else:
                    elem[key] = x[key]
            _data.append(elem)
        for x in _data:
            self._prices.upsert(x,['Code','Date'])

    def downloadCodes(self):
        http = urllib3.PoolManager(32)
        url='https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'
        sheet = pandas.read_excel(url,sheet_name='Sheet1',header=1,usecols=[1,2])
        codes=sheet.values.tolist()
        return codes

    def setCodes(self,codes):
        self._db.begin()
        for code,name in codes:
            data = {
            'Code':code,
            'Name':name,
            }
            self._codes.upsert(data,['Code'])
        self._db.commit()

    def getCodes(self):
        results = self._codes.find()
        data = []
        for record in results:
            data.append(record['Code'])
        return data

    def update(self):
        codes = self.downloadCodes()
        self.setCodes(codes)

    def updateAllStocks(self,years=[2019]):
        self._db.begin()
        for n,code in enumerate(self.getCodes()):
            for year in years:
                try:
                    data = self.downloadStock(code=code,year=year)
                except:
                    print('Exception %d'%code)
                    continue
                self.setStock(data)
                if n%200==0:
                    self._db.commit()
                    print('commit')
                    self._db.begin()
        self._db.commit()

    def downloadDateStock(self,date):
        if not date in self._jpxlist:
            self._jpxlist = self.downloadDateStockList()
            if not date in self._jpxlist:
                raise
        parent,filename = self._jpxlist[date]

        http = urllib3.PoolManager(32)
        url = 'https://www.jpx.co.jp/markets/statistics-equities/daily/{}/{}'.format(parent,filename)
        file = '.x.pdf'
        r = http.request('GET',url,timeout=10.0,retries=10)
        if r.status == 404:
            raise
        with open(file,'wb') as f:
            f.write(r.data)
        stocks ={}
        pages = [(1,)]
        pages.extend(list(it.zip_longest(*[iter(range(2,193))]*10)))
        for page in pages:
            page = list(filter(lambda x:x!=None,page))
            try:
                df = tabula.read_pdf(file,pages=page,guess=True)
                stocks.update(self.pdf_parser(df))
            except:
                for p in page:
                    df = tabula.read_pdf(file,pages=p,guess=True)
                    stocks.update(self.pdf_parser(df))
        return stocks

    def downloadDateStockList(self):
        _urls = [
        'https://www.jpx.co.jp/markets/statistics-equities/daily/index.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-01.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-02.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-03.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-04.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-05.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-06.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-07.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-08.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-09.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-10.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-11.html',
        'https://www.jpx.co.jp/markets/statistics-equities/daily/00-archives-12.html',
        ]
        search = re.compile('stq_[0-9]{8}\.pdf$')
        data = {}
        for url in _urls:
            http = urllib3.PoolManager(32)
            r = http.request('GET',url,timeout=10.0,retries=10)
            if r.status == 404:
                raise
            try:
                soup = BeautifulSoup(r.data,'lxml')
            except:
                soup = BeautifulSoup(r.data,'html.parser')
            table = soup.find('table', class_='overtable')
            for link in table.findAll('a'):
                if search.search(link['href']):
                    x = link['href'].split('/')
                    key = x[-1].replace('stq_','').replace('.pdf','')
                    data[int(key)] = x[-2:]
        return data


    def updateDateStocks(self,date):
        stocks = self.downloadDateStock(date)
        self._db.begin()
        for n,code in enumerate(self.getCodes()):
            if code in stocks:
                data = stocks[code]
                data.update({'Date':int(date)})
                self.setStock([data])
        self._db.commit()

    def pdf_parser(self,df):
        stocks ={}
        for data in df.values:
            if not str(data[0]).isdigit():
                continue
            d = (' '.join(map(str,filter(lambda x:x==x, data)))).replace(',','').split(' ')
            d = list(map(lambda x:x if x!='-' else '0', d))
            while(not d[3].replace('.','').isdigit()):
                d.pop(3)

            code = int(d[0])
            prices = list(map(float,d[3:11]))
            amount = int(float(d[-2])*1000)

            stocks[code]={
            'Code':code,
            'Begin':prices[0] if prices[0]!=0 else prices[4],
            'High':max(prices[1],prices[5]),
            'Low':min(prices[2],prices[6]),
            'End':prices[7] if prices[7]!=0 else prices[3],
            'Amount':amount,
            }
        return stocks



if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-c','--update_code',action='store_true')
    parser.add_argument('-a','--update_all',action='store_true')
    parser.add_argument('-d','--date',type=int,default=20190401)
    args = parser.parse_args()

    d=StockDB()
    if args.update_code:
        d.update()
    if args.update_all:
        d.updateAllStocks()
    else:
        d.updateDateStocks(int(args.date))
