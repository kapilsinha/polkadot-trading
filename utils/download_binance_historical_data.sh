#!/usr/bin/env bash

# This script serially downloads the raw Binance data files and places them in the `data` directory

outdirectory=data/binance_history/glmr_usdt/raw
symbol=GLMRUSDT
startdate=2022-08-01
enddate=2022-09-21

mkdir -p $outdirectory

curr=$startdate
while true; do
    echo "$curr"
    file=$outdirectory/$symbol-aggTrades-$curr.zip
    if [ ! -f "$file" ]
    then
        curl https://data.binance.vision/data/spot/daily/aggTrades/$symbol/$symbol-aggTrades-$curr.zip -o $file
    else
        echo "$file already exists"
    fi
    [ "$curr" \< "$enddate" ] || break
    curr=$( date +%Y-%m-%d --date "$curr +1 day" )
done
