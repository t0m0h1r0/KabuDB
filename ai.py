from analysis import StockDB_Analyzer
import itertools as it


class StockDB_AI(StockDB_Analyzer):
    def getCALC(self,code=6501,date=20190411,term=15):
        param = {
        'Code':code,
        'Date':{'<=':date},
        'order_by':'-Date',
        '_limit':term,
        }
        ret = {}
        for name in self._funcs.keys():
            result = self._db[name].find(**param)
            for x in result:
                y = dict.copy(x)
                date = x['Date']
                del y['id'],y['Code'],y['Date']
                if not (code,date) in ret:
                    ret[(code,date)] = {}
                ret[(code,date)][name]=y
        return ret

    def getParam(self,code=6501,date=20190401,term=15):
        ret = []
        for x in self.getCALC(code=code,date=date,term=term).values():
            z={}
            for y in x.values():
                z.update(y)
            ret.append(list(zip(*sorted(z.items())))[1])
        return ret

a=StockDB_AI()
#print(a.getCALC())
#print(a.getCALC().values())
