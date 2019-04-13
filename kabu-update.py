if __name__ == '__main__':
    from kabu import StockDB_Updater
    import datetime
    import argparse as ap

    today = int('{0:%Y%m%d}'.format(datetime.date.today()))

    parser = ap.ArgumentParser()
    parser.add_argument('-c','--update_code',action='store_true')
    parser.add_argument('-a','--update_all',action='store_true')
    parser.add_argument('-d','--date',type=int,default=today)
    parser.add_argument('-y','--year',type=int,default=2019)
    parser.add_argument('-z','--calc',type=int,default=6501)
    args = parser.parse_args()

    d=StockDB_Updater()
    if args.update_code:
        d.update()
    elif args.update_all:
        d.updateAllStocks(years=[args.year])
    else:
        d.updateDiffStocks()
        #d.updateDateStocks(args.date)
    '''
    a=StockDB_Analyzer()
    print(a.RSI(6081,20190410))
    '''
