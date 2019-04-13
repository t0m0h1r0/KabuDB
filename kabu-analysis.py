if __name__ == '__main__':
    from kabu import StockDB_Analyzer
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-r','--rsi',action='store_true')
    parser.add_argument('-a','--average',action='store_true')
    parser.add_argument('-d','--date',type=int,default=20190401)
    parser.add_argument('-c','--code',type=int,default=6501)
    args = parser.parse_args()

    a=StockDB_Analyzer()
    if args.rsi:
        print(a.RSI(args.code,args.date))
    if args.average:
        print(a.MA25(args.code,args.date),a.MA75(args.code,args.date))
