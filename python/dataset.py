import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

def get_transforms(image_size=128):
    mean = 0.5, 0.5, 0.5
    std = 0.5, 0.5, 0.5
    transform1 = transforms.Compose([transforms.Resize(image_size)])

    transform2 = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform1, transform2


IGNORE_IMAGE = ['Motorofu8jb27smallMotor.jpg' 'Motork3fect0zsmallMotor.jpg'
                'Motorxgebl5t7smallMotor.jpg'
                '2018_10_28_17_16_27_BpfB5fTHAbF_44347259_251926338814290_8879607114401563766_n_1568719865847_18132.jpg'
                'Motore4jrj64tsmallMotor.jpg' 'MotorldccvwphsmallMotor.jpg'
                'Motor6rca1p2nsmallMotor.jpg' 'MotornvhekgknsmallMotor.jpg'
                'Motorbitv4rv0smallMotor.jpg' 'Motor9nh8_1wasmallMotor.jpg'
                'Motorn1u2qdxvsmallMotor.jpg' 'MotorbmzjfsvismallMotor.jpg'
                '2018_09_29_22_57_41_BoU95WmnnD2_42747750_245445649479656_3565246509069139209_n_1568719841783_17808.jpg'
                '2018_02_25_01_09_58_BfmciaeFaXd_28158157_1609026519183202_8836549848904761344_n_1568719652938_15191.jpg'
                'Motorfabw47cmsmallMotor.jpg' 'Motorppxazz94smallMotor.jpg'
                '0780_1568719782781_17003.jpg' 'Motor_8p35he4smallMotor.jpg'
                'Motorta46lisksmallMotor.jpg' 'Motorl2yvl1ozsmallMotor.jpg'
                'Motoreq_60ph9smallMotor.jpg' 'MotorkxdkhqvnsmallMotor.jpg'
                '2018_01_20_18_27_42_BeLmrdcng_N_26320599_317274298767583_7120219748498931712_n_1568719650002_15148.jpg'
                '2019_06_29_15_51_52_BzTKLhZntDb_61433112_456505454914232_3635420167788911772_n_1568719926475_18987.jpg'
                'Motor1_3q4lousmallMotor.jpg' 'Motora1b2zo5ysmallMotor.jpg'
                'Motor8nqn1gsxsmallMotor.jpg' 'MotoromxodgjxsmallMotor.jpg'
                'Motor0gsvincosmallMotor.jpg' 'Motorxut8j1sfsmallMotor.jpg'
                'Motorq09fc573smallMotor.jpg' 'Motorhp5l_uknsmallMotor.jpg'
                '2018_09_22_00_40_54_BoAjWYMHpOO_40741033_329084307638691_8911449314909833615_n_1568719669000_15409.jpg'
                '2018_04_19_23_25_41_BhxTh1bno9W_30855856_397066177424079_1060430689557544960_n_1568719657523_15250.jpg'
                'Motor1a3wniflsmallMotor.jpg'
                '40_1473563140462_5567_1568719177526_8573.jpg'
                'Motoruaalnsv1smallMotor.jpg' 'Motorzk4z555ssmallMotor.jpg'
                'Motorn59c1dg7smallMotor.jpg' 'Motortq4lbb5wsmallMotor.jpg'
                '27_c6b6acf18a12f_032b_1568719409583_11818.jpg'
                '0086_1568719730361_16271.jpg' 'Motorl0n8azgksmallMotor.jpg'
                'd3rdsh_My_First_Bike_M50_Suzuki_LGeoOYh_1568720021078_20169.jpg'
                'Motor14nlqp38smallMotor.jpg' 'Motorp1_4nvdjsmallMotor.jpg'
                'Motorce4mjf_ismallMotor.jpg' 'Motorzsslf7evsmallMotor.jpg'
                'Motorlvbkh6qdsmallMotor.jpg'
                '41_son_xe_sh_son_dan_ao_gia_re_xe_sh_2010_300i_2012_2013_2014_2015_2016_2017_sh_y_2010_sh_300i11667294_779235278862383_1056820945418860794_n_1568719583437_14243.jpg'
                'Motorgk66yavfsmallMotor.jpg' 'Motord4sqou6csmallMotor.jpg'
                'Motorqv9jwhegsmallMotor.jpg'
                '2018_10_02_14_34_37_BobytfWn6sq_42114425_1742876795837753_4936187621724409457_n_1568719847654_17888.jpg'
                'Motorrtsu0pursmallMotor.jpg' 'Motornxprsqn6smallMotor.jpg'
                'Motorftepzdp5smallMotor.jpg' 'Motor0jux8kzdsmallMotor.jpg'
                'Motorwif2gos4smallMotor.jpg' 'Motor0g7f7f5osmallMotor.jpg'
                'Motorllm7hn8msmallMotor.jpg' 'Motorj_s125cmsmallMotor.jpg'
                '2019_02_07_23_46_06_BtmXkrzgsRy_50078492_353323635267219_7230944915609778356_n_1568719890522_18486.jpg'
                '51_GZ150_brochure_red_1568719266666_9806.png'
                'Motor8qerpd2osmallMotor.jpg' '0124_1568719737558_16370.jpg'
                '2018_04_06_01_09_47_BhNcUAelXsl_29717286_2085987974977521_5415278707361185792_n_1568719656418_15235.jpg'
                '2017_11_06_02_54_57_BbI0ULtlhQ2_23279509_1786303208070529_7868756613169938432_n_1568719644049_15062.jpg'
                'Motorgd3r3jq3smallMotor.jpg' 'Motoru_bjc5absmallMotor.jpg'
                'Motorg8snpl03smallMotor.jpg' 'Motorvktog8yusmallMotor.jpg'
                'Motorkokh0ajesmallMotor.jpg' 'Motor4s31xk3csmallMotor.jpg'
                '37_abela_110_5_1568719327443_10655.jpg'
                'cyd54k_Did_some_smaller_upgrades_to_my_MT_07____q4wc6wxot0k31_1568720014870_20109.jpg'
                'Motorjcsbayq0smallMotor.jpg'
                '33_top_nhung_mau_xe_exciter_do_exciter_150_dep_nhat_hien_nay_4_1568719699453_15835.png'
                'Motorgyf41xp9smallMotor.jpg' 'Motor9wiqrjvhsmallMotor.jpg'
                '2019_05_22_09_25_07_BxwntnRAuzT_59649402_411065226413121_6733538138413781868_n_1568719905368_18695.jpg'
                'Motor9fy011qssmallMotor.jpg' 'Motor8s399t14smallMotor.jpg'
                'Motor31cspyybsmallMotor.jpg' 'Motorbi1yaydnsmallMotor.jpg'
                'Motordhynf225smallMotor.jpg' 'Motorqsxidd9nsmallMotor.jpg'
                'Motor380fgvcdsmallMotor.jpg' 'Motorxlgx9c90smallMotor.jpg'
                'Motor7ef8v1t9smallMotor.jpg' 'Motorh3b0ueo2smallMotor.jpg'
                'Motorel38d6l1smallMotor.jpg' 'Motorhvcvip7asmallMotor.jpg'
                'Motorapuwx5kfsmallMotor.jpg' 'Motor481dtxe1smallMotor.jpg'
                'Motor32qbtk95smallMotor.jpg'
                '2019_01_26_00_06_53_BtE7niaAvRX_49627646_413274769410788_6614556006317644289_n_1568719887284_18440.jpg'
                'Motorv0lnjez1smallMotor.jpg'
                '2018_11_19_22_56_16_BqYSRmvACUs_44749290_328343811229511_4303034574978072855_n_1568719673531_15468.jpg'
                'Motor4sjha13psmallMotor.jpg' 'Motor_hyd_eyqsmallMotor.jpg'
                'Motorr4liog42smallMotor.jpg' 'Motorhak794cvsmallMotor.jpg'
                'Motorny8da05fsmallMotor.jpg' 'Motorgk_qncltsmallMotor.jpg'
                'Motor9t9af4rksmallMotor.jpg' 'MotorkmurednssmallMotor.jpg'
                'Motorrmh4549osmallMotor.jpg' 'Motorbaj27hazsmallMotor.jpg'
                '68_thanh_nien_9x_trong_nha_giup_di_tau_ngay_cbr1000_s_a71b_1568719167673_8448.jpg'
                '2018_04_04_23_40_13_BhKtRINlF3J_29715649_188668718410992_5045143871810437120_n_1568719656356_15234.jpg'
                'Motoro86xc2l2smallMotor.jpg'
                '2018_07_31_23_17_09_Bl6gbbgHk6I_37867843_1777210649060248_2379087418679623680_n_1568719665311_15356.jpg'
                'Motor37sr8l8gsmallMotor.jpg' 'Motorj4wnus6msmallMotor.jpg'
                '2018_11_15_05_19_02_BqMGGuhgYhA_44702343_2357632117587270_8042665104017032690_n_1568719873087_18234.jpg'
                '2018_09_29_13_17_24_BoT7fTxn0MT_41607826_704750993235234_2085270240844359457_n_1568719840991_17797.jpg'
                'Motorplhc6y7xsmallMotor.jpg' 'Motork602cg46smallMotor.jpg'
                '2018_11_23_14_05_25_BqhotIjgJxH_44547763_2000958759981571_2282629629862973256_n_1568719875554_18271.jpg'
                'Motorbc1jp7dgsmallMotor.jpg' 'Motorc1c_y9dgsmallMotor.jpg'
                'Motorgewtk7r2smallMotor.jpg'
                '2017_01_06_14_30_06_BO7SNqij_AT_15802231_1410101419040696_7497922509698760704_n_1568719617405_14710.jpg'
                'Motorf49rt3_dsmallMotor.jpg' 'Motor20dbf4q8smallMotor.jpg'
                '62_tem_xe_suzuki_gsx_r150_vang_den_candy13_1568719252520_9610.jpg'
                'Motorrzk6vtuesmallMotor.jpg' 'Motorfjeaopb8smallMotor.jpg'
                '92_xehay_Piaggio_Medley_ABS_test_10__1568719210482_9012.jpg'
                'Motorbrf2kes6smallMotor.jpg'
                '2017_08_29_01_44_07_BYXBW_1hJGX_21107641_113524455995360_2294821027915497472_n_1568719638132_14983.jpg'
                '0167_1568719740550_16413.jpg'
                '49_banner_den_mo_new_1568719224376_9207.png'
                'Motormw55tadosmallMotor.jpg' '78_dwjsyfoju4qyj_1568719315104_10474.jpg'
                'Motorz_u0f8orsmallMotor.jpg' 'Motor_qei95fssmallMotor.jpg'
                '2019_01_26_00_08_22_BtE7ySrAS6G_50095674_124024545315869_759992586016860361_n_1568719887909_18449.jpg'
                '0685_1568719776308_16913.jpg' 'Motoryvdxu7rasmallMotor.jpg'
                '2017_09_12_01_15_45_BY7BPhhHchN_21480442_115905705747252_4489819396599971840_n_1568719639515_14999.jpg'
                'MotorbaxtzhudsmallMotor.jpg'
                '2019_02_07_23_59_54_BtmZJxOAmTP_50292810_2457713457590038_9192013055353439104_n_1568719895556_18556.jpg'
                'Motorb2fv60mmsmallMotor.jpg' 'Motora70fsa5tsmallMotor.jpg'
                'cb6fg6_Can_someone_help_me_identify_this_bike__Its_a_guzzi_I_think_but_dont_know_the_model_or_year_0p34xhru8c931_1568719987671_19814.jpg'
                'Motor92xk2xmssmallMotor.jpg' 'Motorj1gi0998smallMotor.jpg'
                'Motorarp4xfr7smallMotor.jpg' 'Motorhjhebe0vsmallMotor.jpg'
                'Motor39nvrnyxsmallMotor.jpg' 'Motorkfqmab22smallMotor.jpg'
                'Motora_gt_gtlsmallMotor.jpg' 'Motorklqwqj78smallMotor.jpg'
                'Motorveef5yl8smallMotor.jpg'
                '90_20171202_c0fc159fc78aaa6ddc042a427db547b6_1512184724_1568719093525_7405.jpeg'
                'Motor32nqi8tksmallMotor.jpg' 'Motor11yaanzosmallMotor.jpg'
                '2017_05_22_09_28_36_BUY70V0hSeX_18644741_426679091037194_3819485618577080320_n_1568719798603_17226.jpg'
                '2018_09_29_13_35_25_BoT9jOmnvSk_41669106_269107957047935_7621835367417695040_n_1568719841630_17806.jpg'
                '2017_12_02_00_59_55_BcLj0XFFzZR_24254398_1737491536323312_6099541403627421696_n_1568719646195_15094.jpg'
                'Motorj_v73s8usmallMotor.jpg'
                '2019_01_23_23_46_33_Bs_vsvqga5Q_49785037_2305542999729446_5396284773440069596_n_1568719885154_18410.jpg'
                '38_1964_Vespa_90_Small_Frame_made_by_Piaggio_now_in_California_1_1568719074821_7146.jpg'
                'Motor1nkkvbf4smallMotor.jpg' 'Motorjjkbi226smallMotor.jpg'
                'cg2hk9_My_first_ride__1995_750cc_katana__Old_but_still_loving_it__eur5bz00dpb31_1568719995558_19902.jpg'
                'Motorisup45qfsmallMotor.jpg' 'Motor4i5v6_46smallMotor.jpg'
                '0587_1568719769552_16816.jpg' '0823_1568719785641_17044.jpg'
                'Motorltryl31usmallMotor.jpg' 'Motorcl5lvz87smallMotor.jpg'
                'Motori7uvchqwsmallMotor.jpg' 'Motorpstsp3q5smallMotor.jpg'
                '0090_1568719732878_16307.jpg' 'Motorr2lpb2ugsmallMotor.jpg'
                'Motor379vyi6_smallMotor.jpg' 'Motor4t5rda14smallMotor.jpg'
                'Motor6df9g4posmallMotor.jpg' 'Motorqrxnpu91smallMotor.jpg'
                'Motorwtyc3gnxsmallMotor.jpg' 'Motoroscw9t0asmallMotor.jpg'
                '2018_10_28_18_25_27_BpfJy3egjjz_43914322_1975068492584629_5044542269355678902_n_1568719866263_18139.jpg'
                'Motorpyat3iqzsmallMotor.jpg' 'Motorii846bsgsmallMotor.jpg'
                'Motor8cl1e_kysmallMotor.jpg' 'Motorynka93ghsmallMotor.jpg'
                'Motorby708lcrsmallMotor.jpg'
                '2018_11_18_01_30_15_BqTaTsCHgne_44576120_2187749921477744_6137787093380941594_n_1568719673394_15466.jpg'
                '2018_09_29_23_31_20_BoVBv6jHpJp_42004185_1166106346871369_7117847497832332928_n_1568719843046_17826.jpg'
                'Motor84fz2e3msmallMotor.jpg' 'Motor1hf9hhvvsmallMotor.jpg'
                'Motorc0uc9c_1smallMotor.jpg'
                '2018_10_21_03_47_02_BpLjtC_nEaL_43390361_913090252221161_7774329558852033793_n_1568719863314_18096.jpg'
                'Motorb6fu1r_9smallMotor.jpg'
                '2018_09_29_13_50_50_BoT_UJ2nVFV_41579846_169249934003675_331929853950268894_n_1568719841705_17807.jpg'
                '2018_03_26_22_53_01_Bgzctl0FWp6_29402754_356557634855410_8835592762687488000_n_1568719655553_15224.jpg'
                '2018_03_11_22_56_57_BgM1PPEAn6M_28753293_282025522332324_6454297619148570624_n_1568719654188_15209.jpg'
                '2018_02_26_23_12_10_BfrYpS1l7XQ_28435921_1635282323245528_4803966009142673408_n_1568719653079_15193.jpg'
                '2017_02_10_01_30_01_BQUAwuKAYHI_16464939_640945032760355_590732743944437760_n_1568719621148_14759.jpg'
                'Motornkm_ng5qsmallMotor.jpg'
                '77_9e518a34_60e8_ba76_dd00_0cb42b7dbb3b_2019_07_20_00_41_26_0_l_1568719302894_10313.jpg'
                'Motor2haeigossmallMotor.jpg'
                '2018_09_27_04_45_26_BoN3TzMHj_m_41312370_1541853022585817_6418464173870448498_n_1568719824575_17570.jpg'
                'Motorrod_t39xsmallMotor.jpg' 'Motormsh6doilsmallMotor.jpg'
                '2016_12_23_17_09_29_BOXhUrhA2Ib_15623745_945780812222189_2141456181091106816_n_1568719618194_14719.jpg'
                'Motore5lcd4dfsmallMotor.jpg' 'Motorf1k35femsmallMotor.jpg'
                'Motor79vkhln1smallMotor.jpg'
                '2019_06_30_07_39_35_BzU2o0ghc47_65302410_472500136838068_7025238728196995025_n_1568719909133_18749.jpg'
                '2017_10_25_00_58_41_BapteElHO_M_22708849_849050551928731_8627918466722037760_n_1568719643053_15047.jpg'
                'Motorjsi0w40nsmallMotor.jpg' 'Motortj6hue6vsmallMotor.jpg'
                '2018_06_25_23_13_52_Bkdzb5Sn6eR_35252658_2147024111992476_4602625690427195392_n_1568719662697_15319.jpg'
                'cwa07l_CT90_before_and_after_qf54x33vv1j31_1568720015237_20114.jpg'
                '24_angle_hi_bb41_1568719334938_10757.jpg'
                'cmehsq_My_friend_s_first_day_taking_his_new_bike_out__I_think_he_likes_it__6ru364wx6oe31_1568720001232_19969.jpg'
                'MotorisfauqgismallMotor.jpg'
                '2018_11_23_09_33_49_BqhJn3dA48h_43334329_1967242673581018_923517413107514721_n_1568719875417_18269.jpg'
                'Motor4e3ak6xysmallMotor.jpg' 'Motoriqk4l1frsmallMotor.jpg'
                'Motorfzpdvh_1smallMotor.jpg' 'Motor2l9057hesmallMotor.jpg'
                '48_co_nen_lua_chon_yamaha_exciter_2019_hay_honda_winner_2018_bb_baaacVZd9R_1568719437562_12199.png'
                '27_Scoopy_Honda_Launching_2017_DP_5_1568719036225_6602.jpg'
                'Motorw2q6_c3rsmallMotor.jpg' 'Motoruk2jymzysmallMotor.jpg'
                'Motor_een0cxcsmallMotor.jpg' '0664_1568719774795_16892.jpg'
                'Motor2go3zlolsmallMotor.jpg' 'Motorpo8gr1tcsmallMotor.jpg'
                '40_sym_husky_125_classic_19_1568719388562_11530.jpg'
                'MotorjqgcvektsmallMotor.jpg' 'Motorha4cn47dsmallMotor.jpg'
                'Motorn0vt0wwgsmallMotor.jpg' 'Motor5oenxujqsmallMotor.jpg'
                'Motorzmlau9kwsmallMotor.jpg' 'Motors5mu696jsmallMotor.jpg'
                'Motoru1yd2ln2smallMotor.jpg'
                '56_SYM_Star_SR125EFI_do_nhe_tai_hoang_tri_18122017_2_1568719402189_11711.jpg'
                'Motortkwrb1avsmallMotor.jpg' 'Motor266m5ikksmallMotor.jpg'
                'Motoril4z_xeusmallMotor.jpg' 'Motorcz8h74f6smallMotor.jpg'
                'Motordoxi9f6esmallMotor.jpg' 'MotorfppetjxmsmallMotor.jpg'
                '94_426991930_1568719077056_7179.jpg' 'Motorqirc5me5smallMotor.jpg'
                'Motor5j9ry4_vsmallMotor.jpg' 'MotorqrukegbjsmallMotor.jpg'
                'Motornrkqlf1jsmallMotor.jpg' '0531_1568719765815_16761.jpg'
                'Motor9nyeiesmsmallMotor.jpg' 'Motor6e0szksfsmallMotor.jpg'
                'Motoro1niade1smallMotor.jpg' 'Motoruepen9pfsmallMotor.jpg'
                '43_1499009470_1568719303954_10329.jpg' 'Motormqg2d9issmallMotor.jpg'
                'Motoravzdt418smallMotor.jpg' 'Motor65wghhynsmallMotor.jpg'
                '0638_1568719773058_16867.jpg'
                '2017_11_25_07_39_02_Bb6P7dRFihv_23967212_384916131944811_5501735694655029248_n_1568719955320_19377.jpg'
                'Motor9m7d5_2osmallMotor.jpg' 'Motorqzv0p6fysmallMotor.jpg'
                'Motor2hs61vk2smallMotor.jpg' 'MotorxkcoacexsmallMotor.jpg'
                '60_7_800030_1568719714758_16055.jpg' 'Motori5t_zbfvsmallMotor.jpg'
                '38_dscn4253_jpg_7948_1508126155_kwkm_1568719503662_13135.jpg'
                '2019_01_26_00_02_23_BtE7GdzAsP7_49409626_250814415838721_4705127754110376563_n_1568719886863_18434.jpg'
                'cqbp6g_CT90_army_tribute_9x3el90jrfg31_1568720006231_20014.jpg'
                'Motorvmkg5njnsmallMotor.jpg' 'Motorcpt7zeulsmallMotor.jpg'
                'MotorbpkdseygsmallMotor.jpg' 'Motorya6dmoh6smallMotor.jpg'
                'cyrpc3_Riding_in_the_Ozarks_pretty_much_summed_up_in_one_photo__qyfd89sjp7k31_1568720019650_20154.jpg'
                'Motor40b1bogbsmallMotor.jpg' 'Motorlk99nuppsmallMotor.jpg'
                'Motor5f3bufxssmallMotor.jpg'
                '37_20171226_3daad9d63cf6a3d6f46cea07c44a258e_1514298204_1568719104416_7553.jpg'
                'Motor5d4ht13vsmallMotor.jpg' 'Motorq4ria45xsmallMotor.jpg'
                'Motorcvq3r7rusmallMotor.jpg' 'MotorggtppmshsmallMotor.jpg'
                '2018_11_17_16_40_20_BqSdqh4AY9L_44840643_304576673602161_4650684609217623923_n_1568719873829_18245.jpg'
                'Motorjih4ra31smallMotor.jpg' 'Motoriua6xku8smallMotor.jpg'
                'Motor5jyfljcgsmallMotor.jpg' 'Motor6i7e_6xosmallMotor.jpg'
                '0598_1568719770369_16827.jpg' 'Motoryu8hq8wasmallMotor.jpg'
                'Motorvmpvsx2jsmallMotor.jpg' 'Motorv41jbaprsmallMotor.jpg'
                '2018_05_10_04_02_56_BilTJ7yFNoD_31286201_1973596666287790_7005445539311386624_n_1568719808207_17353.jpg'
                'Motorxm9raf91smallMotor.jpg'
                '2018_02_15_23_43_40_BfPHgdBlfTY_27579033_177984302817515_8641469951829344256_n_1568719652266_15181.jpg'
                'Motorp6u26ir1smallMotor.jpg'
                '2018_11_06_23_56_01_Bp26x5ygr0L_44180370_985677628287770_4745427836213439785_n_1568719868787_18175.jpg'
                'Motorvgpcgj0usmallMotor.jpg' 'Motorj71hgw6psmallMotor.jpg'
                'Motorp1qvb1twsmallMotor.jpg' 'Motor2i8h3spksmallMotor.jpg'
                'Motor3jypzs8ksmallMotor.jpg' 'Motorwubfuo82smallMotor.jpg'
                '55_dsc06164jpg_1554830667_1568719059512_6931.jpg'
                'Motorkxbrdns1smallMotor.jpg'
                '38_show_xe_cung_2banh_vn_gz150_rayle_do_phong_cach_hardcore_2234_1389599691_52d39bcbd25d2_1568719265331_9790.jpg'
                '85_xe_ga_1568719580246_14198.jpg' '0380_1568719755503_16626.jpg'
                'Motorroaty2_psmallMotor.jpg' 'Motor8c5_rwohsmallMotor.jpg'
                'Motor4joe9versmallMotor.jpg' 'Motorz49dqe7usmallMotor.jpg'
                'Motorg0fzoksusmallMotor.jpg' 'Motor6lpdfki4smallMotor.jpg'
                'Motor5uv5p3ocsmallMotor.jpg' 'Motort_qsc6f1smallMotor.jpg'
                'MotortaowuxmasmallMotor.jpg' 'Motor8bahh0susmallMotor.jpg'
                'Motormc9mm7pusmallMotor.jpg' 'Motorcfv5041rsmallMotor.jpg'
                '34_31____1557368275_558_width640height480_1568719159074_8331.jpg'
                'Motorhy17u11nsmallMotor.jpg' 'MotorafeoyljesmallMotor.jpg'
                'Motorz1yz4f1ssmallMotor.jpg' 'Motorlyly_f9tsmallMotor.jpg'
                'Motorpvdj1o5rsmallMotor.jpg'
                '2017_11_26_06_39_54_Bb8t9S3nuKd_23969622_1444769625621999_1323476713220341760_n_1568719645737_15087.jpg'
                '2017_11_05_04_22_31_BbGZiphlyKw_23164805_135998543719496_2010156277778350080_n_1568719643977_15061.jpg'
                'Motoro3dk6jz2smallMotor.jpg' 'Motoru7xlml6ksmallMotor.jpg'
                'Motormnypb0wqsmallMotor.jpg' 'Motorjp975mnnsmallMotor.jpg'
                'Motora70j7rd9smallMotor.jpg' 'Motordsu474prsmallMotor.jpg'
                'Motorr4qan93_smallMotor.jpg' 'Motord1xgsebksmallMotor.jpg'
                'Motor5zph81gvsmallMotor.jpg' 'Motorpn7t4icgsmallMotor.jpg'
                'Motorvzwdtxy2smallMotor.jpg' 'Motor2_t59p7nsmallMotor.jpg'
                'Motor_l11__0dsmallMotor.jpg'
                '2019_01_26_00_13_14_BtE8V9mgW69_49308636_957393307799148_1890423269476442725_n_1568719888875_18463.jpg'
                'Motor7vbtz86wsmallMotor.jpg' 'Motorr2zyagt9smallMotor.jpg'
                'Motorlz6sqtn6smallMotor.jpg'
                '2017_10_03_01_12_10_BZxFhoiHLZ2_18949986_695287564010258_6044590501120704512_n_1568719641312_15023.jpg'
                'Motor5k318m4usmallMotor.jpg' 'Motor8f_ztiftsmallMotor.jpg'
                'Motordaxix3s_smallMotor.jpg'
                '2018_11_11_01_12_26_BqBWtGzlSGO_44585918_261894171183781_147229160666285630_n_1568719672921_15459.jpg'
                'Motors75lk7ncsmallMotor.jpg'
                '96_20180104_suzuki_khuyen_mai_tet_mau_xe_gd110hu_gia_tre_me_man_1_1568719243336_9474.jpg'
                '2017_06_18_01_24_56_BVdl7goFfJh_19228451_1263572657095654_3317026834956156928_n_1568719630710_14888.jpg'
                '2018_08_14_23_04_30_BmeiHG7HNab_38738849_525257354601964_3264219301518770176_n_1568719666272_15371.jpg'
                'Motorrptlu757smallMotor.jpg' 'Motor9pq200czsmallMotor.jpg'
                'Motorx69okntlsmallMotor.jpg' 'Motorsz7q3_65smallMotor.jpg'
                'Motor9r9dbtr0smallMotor.jpg' 'Motor9cvpy2h3smallMotor.jpg'
                'Motor6y1t_9xqsmallMotor.jpg'
                '62_078420224b63a23dfb72_1568719108055_7606.jpg'
                'Motorqm592koismallMotor.jpg' 'Motor5s8l6eqwsmallMotor.jpg'
                '2019_01_21_08_02_00_Bs46A_Sgp1o_49663057_2328839980519266_8881828202368531467_n_1568719883854_18391.jpg'
                '2018_01_24_07_39_26_BeUvq8vFQ0W_26865108_357628584711010_6659422237843521536_n_1568719959674_19439.jpg'
                'Motorv42qy8eusmallMotor.jpg' 'Motorm0bmounesmallMotor.jpg'
                'Motorqt7mnn_9smallMotor.jpg' 'Motorcqk_zkorsmallMotor.jpg'
                'MotoryyylhzfasmallMotor.jpg' 'Motoroln38t7jsmallMotor.jpg'
                'Motorqpup6djusmallMotor.jpg' 'Motor2tvlyfa9smallMotor.jpg'
                'Motorg9k9yt6msmallMotor.jpg' 'Motorjt9k0r0lsmallMotor.jpg'
                'Motorj09inqsesmallMotor.jpg' 'Motor688u2zyismallMotor.jpg'
                'Motor82lhdtaesmallMotor.jpg' 'Motor39po6fpvsmallMotor.jpg'
                'Motor1tlkud2vsmallMotor.jpg' 'Motoroulj5am5smallMotor.jpg'
                'Motorruye00ftsmallMotor.jpg' 'MotorznbvbvnismallMotor.jpg'
                'Motorv4oucsf0smallMotor.jpg' 'Motoriaj0d_swsmallMotor.jpg'
                '2017_11_26_01_08_14_Bb8IADUlm13_23825167_160357941377753_4889945748260519936_n_1568719645668_15086.jpg'
                '2018_10_10_00_36_03_Bou5GnhH2ob_42839855_139602043663823_3469477301036828174_n_1568719670239_15427.jpg'
                '56_tem_xe_suzuki_gsx_750_xanh_gp6_1568719500643_13097.jpg'
                'Motor2fankuyqsmallMotor.jpg' 'Motorz9bazt6jsmallMotor.jpg'
                'Motor54iej6m3smallMotor.jpg' 'Motorbf2j67a2smallMotor.jpg'
                '30_3e8bd1a3949be86a06cfa49bb8fe5b06_2629988363311584725_1568719090481_7362.jpg'
                'Motorf3k3z8y2smallMotor.jpg' 'Motors0bfhl9qsmallMotor.jpg'
                'cg97uv_My_first_long_ride_from_Manila_to_Bicol___438_km__vc873u96lsb31_1568719996085_19909.jpg'
                'Motora80j37lgsmallMotor.jpg'
                '59_vulcanreg_900_classic_lt52_1568719107411_7598.jpg'
                '73_55a9ba5688e841437186646_1568719349991_10973.jpg'
                'Motor2c6w2is6smallMotor.jpg' 'Motor34173armsmallMotor.jpg'
                'Motorbz1zif6dsmallMotor.jpg' 'Motort_5jg56_smallMotor.jpg'
                'Motorhmf3j3c6smallMotor.jpg'
                '2017_12_27_22_50_45_BdORtLFBcNO_25023452_1949871961996240_6511310303525863424_n_1568719804189_17297.jpg'
                'Motorz9qpdwyusmallMotor.jpg' 'Motoratm0hkg9smallMotor.jpg'
                'Motor7u7p2cchsmallMotor.jpg' 'Motor6l5qp_kfsmallMotor.jpg'
                'Motora6vn8kh8smallMotor.jpg' 'Motorhp6cy6nfsmallMotor.jpg'
                '2017_05_30_01_28_28_BUsrOyaF0Tm_18722476_163800680824271_884625334396256256_n_1568719629403_14869.jpg'
                'Motorc01b7owismallMotor.jpg' 'Motor150akxgismallMotor.jpg'
                'Motorggdjgz1vsmallMotor.jpg' 'Motor8dvcwvz5smallMotor.jpg'
                '2017_02_15_01_01_16_BQg1cgGDV_Y_16585708_281707652248409_5505754018812526592_n_1568719621503_14764.jpg'
                'Motorl4a2qd80smallMotor.jpg' 'Motor2nzxvygzsmallMotor.jpg'
                'Motorpdva2zq0smallMotor.jpg' 'MotoryingckazsmallMotor.jpg'
                'MotorelopfttxsmallMotor.jpg'
                '2019_01_26_00_03_31_BtE7Ox7FS4A_50231711_103796537408968_2339202674174277448_n_1568719679177_15546.jpg'
                'Motorlc2prpw0smallMotor.jpg'
                'cr10hl_Wandering_in_the_mountains__taking_it_easy_on_the_twisties___kybczd21iqg31_1568720007050_20026.jpg'
                'Motorkchinkn7smallMotor.jpg' 'Motor290_7wd0smallMotor.jpg'
                'Motor5pbbrme2smallMotor.jpg' 'Motorj_s9i773smallMotor.jpg'
                'Motorsbxml7u3smallMotor.jpg' 'Motor78hwe9ldsmallMotor.jpg'
                'Motoruboe0prbsmallMotor.jpg' 'Motor3lb_49fksmallMotor.jpg'
                'Motorq6t8fs7dsmallMotor.jpg' 'Motorz0pyspi9smallMotor.jpg'
                'Motor06ya1g5msmallMotor.jpg' 'Motorol4n2_ewsmallMotor.jpg'
                'Motor5tutpn1rsmallMotor.jpg'
                '2018_10_13_10_55_06_Bo3uVRTn7sk_42324145_2047815331943459_3902440884092273142_n_1568719856852_18006.jpg'
                'Motoryp3ur362smallMotor.jpg' 'Motorcom0b3stsmallMotor.jpg'
                'Motorhj1lj5zgsmallMotor.jpg' 'Motorhbut3_mvsmallMotor.jpg'
                'Motor0md9koh_smallMotor.jpg' 'Motorgizbcj_4smallMotor.jpg'
                '2018_07_22_22_57_09_BljS_KDnHh4_37052799_273726046735278_69344488966848512_n_1568719664597_15346.jpg'
                'Motortl9lrqsbsmallMotor.jpg'
                '2018_06_17_02_02_56_BkG7oXGn8zV_33649695_176269236386549_7298684006244024320_n_1568719662040_15310.jpg'
                '13_1507015309_150695333767367_honda1_1568719004869_6168.jpg'
                'Motor9k1djzy3smallMotor.jpg' 'Motor4k0q2_ipsmallMotor.jpg'
                'Motorf7tw5b0psmallMotor.jpg' 'Motorsjnh3zbismallMotor.jpg'
                'Motork4jnholbsmallMotor.jpg' 'Motora2it9zijsmallMotor.jpg'
                '0107_1568719736402_16352.jpg' 'Motorhbuyped1smallMotor.jpg'
                'Motornvu73c7rsmallMotor.jpg' 'Motor7twf44vxsmallMotor.jpg'
                '21_1547536060_vatgia2_picture6859_1568719083108_7265.jpg'
                'Motorg8aar7irsmallMotor.jpg' 'Motor840s58mrsmallMotor.jpg'
                '9_5_1_1568719592744_14365.png'
                '2017_04_02_23_24_25_BSZruT_hEyE_17662294_1470384943032548_4517400186221232128_n_1568719624961_14811.jpg'
                'Motorz270p4j0smallMotor.jpg' 'Motor_q02y_ilsmallMotor.jpg'
                'Motor7p91bqs1smallMotor.jpg' 'Motorr84bt33jsmallMotor.jpg'
                'Motorvmwl3v8gsmallMotor.jpg' 'Motor3wh0b72dsmallMotor.jpg'
                '18_520_368_0c64d29bf3e5621b5db15279e2725139_1568719486480_12898.jpg'
                'Motor88va8j9usmallMotor.jpg'
                '2018_02_12_23_48_07_BfHZocwF6Vr_27574430_746454952215612_7977680308101185536_n_1568719652011_15177.jpg'
                'Motorkm52mp_wsmallMotor.jpg' 'Motor182cktm8smallMotor.jpg'
                'MotorqxtcythjsmallMotor.jpg' 'Motor7yxp6znusmallMotor.jpg'
                'Motor3zkv_r50smallMotor.jpg' 'Motorc3h0wblcsmallMotor.jpg'
                'Motortm11rh9ksmallMotor.jpg' 'Motorbc8ci39usmallMotor.jpg'
                'MotorthjgpylqsmallMotor.jpg' 'Motorscypx2b7smallMotor.jpg'
                'Motorb6xpprefsmallMotor.jpg'
                '2018_10_29_11_57_06_BphCJWen_bN_43778785_1131872193646259_1379171257112946434_n_1568719866750_18146.jpg'
                'Motorj_yrddqwsmallMotor.jpg' 'Motoromh4r8fosmallMotor.jpg'
                'Motorixm7211xsmallMotor.jpg' 'Motorkkja94oqsmallMotor.jpg'
                'Motor0g13l15gsmallMotor.jpg' 'Motor909v_v1ismallMotor.jpg'
                '2017_02_14_01_00_09_BQeQhfdgLfC_16464157_1791795671141501_7457650370890694656_n_1568719621430_14763.jpg'
                'Motorhlzjo1yasmallMotor.jpg'
                '2017_07_07_00_04_48_BWOX3MoFlCX_19932018_1094425803991922_3757391876219469824_n_1568719632377_14910.jpg'
                'Motorple2bq95smallMotor.jpg' 'Motorhllso7b4smallMotor.jpg'
                'Motork73zr00ismallMotor.jpg'
                '2018_05_13_22_18_29_Biu_6mnnJrT_31502946_1882402145391395_37023322466156544_n_1568719659379_15275.jpg'
                'Motor7nc_7gg9smallMotor.jpg' 'Motoriib0rc2_smallMotor.jpg'
                'Motorn27nqsiusmallMotor.jpg' 'Motor54d87zbzsmallMotor.jpg'
                '2017_06_08_01_24_45_BVD19d_FKZe_18888516_2003395763222669_987425445345492992_n_1568719630044_14878.jpg'
                'Motorgelqf84rsmallMotor.jpg' 'Motore7jt5u1ksmallMotor.jpg'
                'Motor5s7oq9zismallMotor.jpg' 'Motorhgfuy9e5smallMotor.jpg'
                '2018_08_09_22_57_22_BmRpUZinbrA_37868578_1891228720920548_8760269434431471616_n_1568719665932_15366.jpg'
                'Motor4rf9chnysmallMotor.jpg' 'Motord66y0xjgsmallMotor.jpg'
                'Motora3g6ihj0smallMotor.jpg' '94_9082748988_1568719268930_9839.jpg'
                '2017_09_27_01_26_30_BZhqZaJH0m__21911361_150689848863720_7700300423793475584_n_1568719640819_15016.jpg'
                'Motork7gfx5fgsmallMotor.jpg'
                '2018_11_25_22_43_55_BqntoYZlhkK_44777166_1952933695016209_1399861106154354720_n_1568719673935_15474.jpg'
                'Motor4fm1x0mcsmallMotor.jpg' 'Motor46y7r_9vsmallMotor.jpg'
                '2018_11_22_04_31_34_BqeCPKSFA2X_43698083_707119929671825_3456215231687032832_n_1568719922449_18928.jpg'
                'Motorb11_7sc1smallMotor.jpg'
                '2018_01_20_01_18_35_BeJw5_2Fksu_26181063_2147518368809346_376029467711111168_n_1568719649932_15147.jpg'
                'Motordh558dw5smallMotor.jpg'
                '2018_09_27_11_50_52_BoOn_scDzx__41863391_335511580326950_3178184615330240457_n_1568719826887_17599.jpg']


class MotorbikeDataset(Dataset):
    def __init__(self, path, transform1=None, transform2=None,
                 ignore=None):
        if ignore is None:
            ignore = IGNORE_IMAGE
        self.path = path
        img_list = os.listdir(self.path)
        img_list = [i for i in img_list if i not in ignore]
        self.img_list = img_list
        self.transform1 = transform1
        self.transform2 = transform2
        self.load_data(img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform2 is not None:
            img = self.transform2(img)
        return img

    def load_data(self, img_list):
        self.images = []
        for idx, path in enumerate(img_list):
            origin_img = Image.open(os.path.join(self.path, self.img_list[idx]))
            img = origin_img.copy()
            origin_img.close()
            img = self.transform1(img)
            self.images.append(img)
        return self.images

    def sample(self, n=5):
        return [self.__getitem__(i) for i in range(n)]


class MultipleResMotorbikeDS(Dataset):
    def __init__(self, path,
                 transform1=None,
                 transform2=None,
                 ignore=None,
                 resolution=128):
        if ignore is None:
            ignore = IGNORE_IMAGE
        self.path = path
        self.resolution = resolution
        img_list = os.listdir(self.path)
        img_list = [i for i in img_list if i not in ignore]
        self.img_list = img_list[:32]

        if transform1 is None and transform2 is None:
            self.transform1, self.transform2 = get_transforms()
        else:
            self.transform1 = transform1
            self.transform2 = transform2
        self.reload_data(resolution)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.images[idx]

        if self.transform2 is not None:
            img = self.transform2(img)
        return img

    def reload_data(self, resolution):
        # update transformer
        self.transform1, _ = get_transforms(resolution)
        img_list = self.img_list
        # reload all data
        self.images = []
        import gc;
        gc.collect()
        for idx, path in enumerate(img_list):
            origin_img = Image.open(os.path.join(self.path, self.img_list[idx]))
            img = origin_img.copy()
            origin_img.close()
            img = self.transform1(img)
            self.images.append(img)
        return self.images

    def sample(self, n=5):
        return [self.__getitem__(i) for i in range(n)]


class MotorbikeWithLabelsDataset(Dataset):
    def __init__(self, path, labels,
                 transform1=None, transform2=None,
                 ignore=None, in_memory=True):
        if ignore is None:
            ignore = IGNORE_IMAGE
        self.path = path
        img_list = os.listdir(self.path)
        img_list = [i for i in img_list if i not in ignore]
        self.img_list = img_list
        self.transform1 = transform1
        self.transform2 = transform2

        get_fname = lambda x: '.'.join(x.split('.')[:-1])
        label_path = labels
        label_df = pd.read_csv(label_path)
        label_df['image_id'] = label_df['image_id'].apply(get_fname)
        label_df = label_df.set_index('image_id')
        self.class_dist = label_df['class'].value_counts().sort_index().values / float(len(label_df))
        self.mapping_id_label = label_df['class'].to_dict()
        self.in_memmory = in_memory
        # filter the image list for safe
        self.img_list = [i for i in self.img_list if get_fname(i) in
                         self.mapping_id_label]

        if self.in_memmory:
            self.load_data(img_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        id = '.'.join(img_name.split('.')[:-1])
        if self.in_memmory:
            img = self.images[idx]
        else:
            origin_img = Image.open(os.path.join(self.path, self.img_list[idx]))
            img = origin_img.copy()
            origin_img.close()
            img = self.transform1(img)

        if self.transform2 is not None:
            img = self.transform2(img)

        return img, self.mapping_id_label[id]

    def load_data(self, img_list):
        self.images = []
        for idx, path in enumerate(img_list):
            origin_img = Image.open(os.path.join(self.path, self.img_list[idx]))
            img = origin_img.copy()
            origin_img.close()
            img = self.transform1(img)
            self.images.append(img)
        return self.images

    def get_class_distributions(self):
        return self.class_dist

    def sample(self, n=5):
        return [self.__getitem__(i) for i in range(n)]

if __name__ == '__main__':
    tf1, tf2 = get_transforms(128)
    ds = MotorbikeWithLabelsDataset('../data/resized128_image_fixed', '../data/label.csv',
                                    tf1, tf2,
                                    in_memory=False)
    print(ds.get_class_distributions())
    # print(ds[1])
    # print(ds.img_list[1])
    # for i in range(len(ds)):
    #     try:
    #         _ = ds[i]
    #     except:
    #         print(f"{i}")