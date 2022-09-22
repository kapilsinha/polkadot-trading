#!/usr/bin/env bash

# This script serially downloads the raw OKX data files and places them in the `data` directory

outdirectory=data/okx_history/dot_usdt/raw
symbol=DOT-USDT
# DOT-USDT
startdate=2022-08-01
enddate=2022-09-21

mkdir -p $outdirectory

curr=$startdate
while true; do
    echo "$curr"
    file=$outdirectory/$symbol-aggTrades-$curr.zip
    if [ ! -f "$file" ]
    then
        nohyphencurr=$(echo $curr | tr -d '-')
        curl https://static.okx.com/cdn/okex/traderecords/aggtrades/daily/$nohyphencurr/$symbol-aggtrades-$curr.zip -o $file
    else
        echo "$file already exists"
    fi
    [ "$curr" \< "$enddate" ] || break
    curr=$( date +%Y-%m-%d --date "$curr +1 day" )
done
