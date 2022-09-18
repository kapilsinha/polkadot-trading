#!/usr/bin/env bash

# This script serially downloads the raw Binance data files and places them in the `data` directory

outdirectory=data/binance_history/avax_usdt/raw
symbol=AVAXUSDT
startdate=2022-08-01
enddate=2022-09-15

mkdir -p $outdirectory

curr=$startdate
while true; do
    echo "$curr"
    curl https://data.binance.vision/data/spot/daily/aggTrades/$symbol/$symbol-aggTrades-$curr.zip -o $outdirectory/$symbol-aggTrades-$curr.zip 
    # unzip raw/$symbol-aggTrades-$curr.zip -d raw # pandas reads .zip files so we leave as-is
    [ "$curr" \< "$enddate" ] || break
    curr=$( date +%Y-%m-%d --date "$curr +1 day" )
done
