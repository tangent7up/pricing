# -*- coding:utf-8 -*-
path = '/home/ymserver/workplace/rtb_online/'
auto_search_ad = True
ad = [114333,114368,114290]


bid_path = '/data/bid/producttype=*/date=2017-09-1[4-5]'
bid_session_path = '/data/session/date=2017-09-1[4-5]'
session_paths = [
'/data/session/date=2017-09-{0[0-9]}',
'/data/session/date=2017-08-{2[0-9],3[0-1]}',
'/data/session/date=2017-08-{1[0-9]}',
'/data/session/date=2017-08-{0[1-9]}',
'/data/session/date=2017-07-{2[0-9],3[0-1]}',
'/data/session/date=2017-07-{1[0-9]}',
'/data/session/date=2017-07-{0[1-9]}',
'/data/session/date=2017-06-{2[0-9],30}',
'/data/session/date=2017-06-{1[0-9]}',
'/data/session/date=2017-06-{0[1-9]}'
]
#session_old_path = '/data/session/date=2017-{08-{1[8-9],2[0-9],3[0-1]},09-{0[0-9],1[0-3]}}'
session_old_path = '/data/session/date=2017-{05-{2[4-9],3[0-1]},06-*,07-*,08-*,09-{0[1-9],1[0-9],2[0-2]}}'
renew_bid_constand = True


# bid_path = '/data/bid/producttype=*/date=2017-09-{0[3-4]}'
# session_path = '/data/session/date=2017-09-{0[3-4]}'
# session_old_path = '/data/session/date=2017-09-{0[1-2]}'





# path = './'
# auto_search_ad = False
# ad = [114333,114368,114290]
# bid_path = 'bid'
# bid_session_path = 'win'
# session_path = 'win'
# session_old_path = 'win'
#
# renew_bid_constand = True
