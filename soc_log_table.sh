#!/bin/sh
# Convert SOC prediction + Magic SOC log lines to a unified markdown table.
# Usage: ./soc_log_table.sh /tmp/hass.log

FILE="${1:?usage: $0 <logfile>}"

(
  # Charging prediction (deduplicated by timestamp+VIN)
  grep 'SOC: Predicted' "$FILE" |
    sed 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Predicted \([0-9.]*%\) for \([^ ]*\) (anchor=\([0-9.]*%\), +\([0-9.]* kWh\) net, eff=\([0-9]*%\)).*/\1|\3|Charging|\2 (anchor \4, +\5, \6)/' |
    sort -t'|' -k1,2 -u

  # Charging power + anchor + sync events
  grep -E 'Power update|SOC:.*Anchored session|SOC:.*Charging started|SOC:.*Charging stopped|SOC:.*sync' "$FILE" |
    sed \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Power update for \([^ ]*\): \(.*\)/\1|\2|Power|\3/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Anchored session for \([^ ]*\) at \(.*\)/\1|\2|Charge anchor|\3/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Charging started for \([^ ]*\)/\1|\2|Charge start|/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Charging stopped for \([^ ]*\)/\1|\2|Charge stop|/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* BEV \([^ ]*\) .* BMW SOC \(.*\)/\1|\2|BMW sync|\3/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* PHEV \([^ ]*\) .* BMW SOC \(.*\)/\1|\2|BMW sync|\3/' |
    grep '|'

  # Driving sessions (anchor, re-anchor, end, learning)
  grep 'Magic SOC:' "$FILE" |
    grep -E 'Anchor|Re-anchor|end_driving|learn|consumption' |
    grep -v 'Skipping anchor\|Set default\|reset button\|Created' |
    sed \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Anchored driving session for \([^ ]*\) at \([0-9.]*\)% \/ \([0-9.]*\) km (consumption=\([0-9.]*\) kWh\/km)/\1|\2|Drive anchor|\3% at \4 km (cons=\5)/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Re-anchored \([^ ]*\) from \([0-9.]*\)% to \([0-9.]*\)% at \([0-9.]*\) km/\1|\2|Re-anchor|\3% -> \4% at \5 km/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* end_driving_session for \([^ ]*\) but no active session/\1|\2|Drive end|no active session/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Skipping learning for \([^ ]*\): \(.*\)/\1|\2|Skip learn|\3/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Learning for \([^ ]*\): \(.*\)/\1|\2|Learned|\3/' \
      -e 's/.*\([0-9][0-9]:[0-9][0-9]:[0-9][0-9]\)\.[0-9]*.* Ended driving session for \([^ ]*\): \(.*\)/\1|\2|Drive end|\3/' |
    grep '|'
) | sort -t'|' -k1,1 |
awk -F'|' 'BEGIN {
  print "| Time | VIN | Event | Details |"
  print "|------|-----|-------|---------|"
}
{ printf "| %s | %s | %s | %s |\n", $1, $2, $3, $4 }'
