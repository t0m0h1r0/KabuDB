if __name__ == '__main__':
    from kabu import StockDB_Analyzer
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
    for code in args.codes:
        print('Code:',code)
        if args.rsi:
            print('RSI: ',a.RSI(code,args.date))
        if args.average:
            print('Ave: ',a.Average(code,args.date))
