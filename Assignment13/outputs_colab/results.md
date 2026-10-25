| experiment | parameters | variant | seq | micro-batch | effective batch | train tokens | train loss | val loss | tok/s | wall (s) | peak alloc MiB | peak reserved MiB | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 84608 | none | 32 | 4 | 4 | 512 | 5.508 | 5.485 | 5779 | 0.09654 | n/a (cpu) | n/a (cpu) | smoke |
| B | 84608 | euler | 32 | 4 | 4 | 512 | 5.516 | 5.498 | 8366 | 0.06693 | n/a (cpu) | n/a (cpu) | smoke |
| B | 84608 | midpoint | 32 | 4 | 4 | 512 | 5.517 | 5.501 | 7742 | 0.07085 | n/a (cpu) | n/a (cpu) | smoke |
| C | 84608 | euler | 32 | 8 | 8 | 1024 | 5.502 | 5.474 | 5084 | 0.2078 | n/a (cpu) | n/a (cpu) | smoke |
