#!/bin/sh
# Convert SOC prediction log lines to a markdown table.
# Usage: ./soc_log_table.sh /tmp/hass.log

grep 'SOC: Predicted' "${1:?usage: $0 <logfile>}" |
  sed 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]* .* Predicted \([0-9.]*%\).*(anchor=\([0-9.]*%\), +\([0-9.]* kWh\) net, eff=\([0-9]*%\)).*/\1|\2|\3|\4|\5/' |
  sort -t'|' -k1,1 -u |
  awk -F'|' 'BEGIN {
    print "| Time | Predicted | Anchor | Net energy | Efficiency |"
    print "|------|-----------|--------|------------|------------|"
  }
  { printf "| %s | %s | %s | +%s | %s |\n", $1, $2, $3, $4, $5 }'
