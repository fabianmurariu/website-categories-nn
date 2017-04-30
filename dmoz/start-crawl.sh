#!/usr/bin/env bash
# assume the file dmoz_domain_category.tab is in the local directory
unxz dmoz_domain_category.tab.xz
split --bytes=10M dmoz_domain_category.tab

if [ -f /proc/cpuinfo ]; then
    export CPUS=$(grep -c ^processor /proc/cpuinfo)
else
    export CPUS=$(sysctl -n hw.ncpu)
fi

ls x* | xargs -n 1 -P ${CPUS} ./crawl-file.sh $1