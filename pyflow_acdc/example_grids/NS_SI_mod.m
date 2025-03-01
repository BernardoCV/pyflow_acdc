function mpc = NS_SI_mod()

mpc.version = '2';

mpc.baseMVA = 100;

%% AC bus data
%    bus_i    type    Pd    Qd    Gs    Bs    area    Vm    Va    baseKV    zone    Vmax    Vmin
mpc.bus = [
   1     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   2     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   3     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   4     3     0.0     0.0     0.0     0.0     2     1.0     0.0     380     2     1.1     0.9;
   5     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   6     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   7     1     1300.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   8     1     550.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   9     2     560.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   10     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   11     1     515.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   12     1     520.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   13     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   14     1     1620.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   15     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   16     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   17     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   18     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   19     2     120.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   20     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   21     2     700.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   22     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   23     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   24     2     340.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   25     3     0.0     0.0     0.0     0.0     1     1.0     0.0     380     1     1.1     0.9;
   26     2     575.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   27     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   28     1     1320.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   29     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   30     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   31     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   32     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   33     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   34     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   35     1     800.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   36     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   37     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   38     2     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   39     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   40     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   41     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   42     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   43     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   44     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   45     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   46     1     6280.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   47     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   48     2     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   49     1     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   50     2     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   51     2     0.0     0.0     0.0     0.0     1     1.01     0.01     380     1     1.1     0.9;
   52     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   53     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   54     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   55     1     710.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   56     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   57     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   58     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   59     1     930.0000000000001     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   60     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   61     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   62     1     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   63     2     0.0     0.0     0.0     0.0     2     1.01     0.01     380     2     1.1     0.9;
   64     2     0.0     0.0     0.0     0.0     3     1.01     0.01     380     3     1.1     0.9;
   65     2     0.0     0.0     0.0     0.0     3     1.01     0.01     380     3     1.1     0.9;
   66     2     0.0     0.0     0.0     0.0     3     1.01     0.01     380     3     1.1     0.9;
   67     1     0.0     0.0     0.0     0.0     3     1.01     0.01     380     3     1.1     0.9;
   68     1     560.0     0.0     0.0     0.0     3     1.01     0.01     380     3     1.1     0.9;
   69     1     0.0     0.0     0.0     0.0     3     1.01     0.01     380     3     1.1     0.9;
   70     1     0.0     0.0     0.0     0.0     3     1.01     0.01     380     3     1.1     0.9;
   71     3     240.0     0.0     0.0     0.0     3     1.0     0.0     380     3     1.1     0.9;
   72     2     0.0     0.0     0.0     0.0     3     1.01     0.01     380     3     1.1     0.9;
   73     3     0.0     0.0     0.0     0.0     4     1.0     0.0     380     4     1.0     1.0;
   74     3     0.0     0.0     0.0     0.0     5     1.0     0.0     380     5     1.0     1.0;
   75     3     0.0     0.0     0.0     0.0     6     1.0     0.0     380     6     1.0     1.0;
   76     3     0.0     0.0     0.0     0.0     7     1.0     0.0     380     7     1.0     1.0;
   77     3     0.0     0.0     0.0     0.0     8     1.0     0.0     380     8     1.0     1.0;
];

%% AC branch data
%    fbus    tbus    r    x    b    rateA    rateB    rateC    ratio    angle    status    angmin    angmax
mpc.branch = [
    25     26     0.00017313019390581717     0.0023202967902032647     0.30621103594539706     4680.0     4680.0     4680.0     0     0     1     -360     360;
    25     27     0.0005817174515235457     0.007796197215082969     4.115476323106137     9359.0     9359.0     9359.0     0     0     1     -360     360;
    27     28     0.0001592797783933518     0.002134673046987004     0.28171415306976527     4680.0     4680.0     4680.0     0     0     1     -360     360;
    27     30     0.00034626038781163435     0.004640593580406529     0.6124220718907941     4680.0     4680.0     4680.0     0     0     1     -360     360;
    28     29     7.617728531855956e-05     0.0010209305876894366     0.13473285581597472     4680.0     4680.0     4680.0     0     0     1     -360     360;
    29     30     0.000110803324099723     0.0014849899457300894     0.19597506300505413     4680.0     4680.0     4680.0     0     0     1     -360     360;
    30     31     0.0002631578947368421     0.003526851121108963     0.46544077463700356     4680.0     4680.0     4680.0     0     0     1     -360     360;
    31     32     0.00035422437673130196     0.0012814391087010999     7.875294198536438     2633.0     2633.0     2633.0     0     0     1     -360     360;
    32     33     0.0004951523545706371     0.001791258969152075     6.192267615784699     1975.0     1975.0     1975.0     0     0     1     -360     360;
    30     34     5.54016620498615e-05     0.0007424949728650447     0.39195012601010826     9359.0     9359.0     9359.0     0     0     1     -360     360;
    34     35     0.00045706371191135735     0.00612558352613662     0.8083971348958483     4680.0     4680.0     4680.0     0     0     1     -360     360;
    35     39     0.0006925207756232687     0.009281187160813059     1.2248441437815882     4680.0     4680.0     4680.0     0     0     1     -360     360;
    34     40     0.001066481994459834     0.014293028227652112     1.886259981423646     4680.0     4680.0     4680.0     0     0     1     -360     360;
    34     41     0.001066481994459834     0.014293028227652112     1.886259981423646     4680.0     4680.0     4680.0     0     0     1     -360     360;
    34     36     0.0003739612188365651     0.005011841066839052     0.6614158376420577     4680.0     4680.0     4680.0     0     0     1     -360     360;
    36     37     0.00030470914127423825     0.004083722350757746     0.5389314232638989     4680.0     4680.0     4680.0     0     0     1     -360     360;
    37     42     0.0006024930747922437     0.008074632829907362     1.0656144050899818     4680.0     4680.0     4680.0     0     0     1     -360     360;
    37     38     0.0005609418282548476     0.007517761600258578     0.9921237564630865     4680.0     4680.0     4680.0     0     0     1     -360     360;
    38     44     0.00036703601108033245     0.004919029195230921     0.6491673962042418     4680.0     4680.0     4680.0     0     0     1     -360     360;
    39     40     0.00019390581717451525     0.0025987324050276567     0.34295636025884474     4680.0     4680.0     4680.0     0     0     1     -360     360;
    40     41     0.000110803324099723     0.0014849899457300894     0.19597506300505413     4680.0     4680.0     4680.0     0     0     1     -360     360;
    41     42     0.00017313019390581717     0.0023202967902032647     0.30621103594539706     4680.0     4680.0     4680.0     0     0     1     -360     360;
    42     44     0.00046398891966759005     0.006218395397744749     0.8206455763336642     4680.0     4680.0     4680.0     0     0     1     -360     360;
    42     43     0.00047091412742382275     0.0063112072693528795     0.20822350444287002     2340.0     2340.0     2340.0     0     0     1     -360     360;
    43     44     0.0004293628808864266     0.0057543360397040964     0.18985084228614618     2340.0     2340.0     2340.0     0     0     1     -360     360;
    39     45     0.00017313019390581717     0.0023202967902032647     0.30621103594539706     4680.0     4680.0     4680.0     0     0     1     -360     360;
    45     47     0.0003116343490304709     0.0041765342223658766     0.5511798647017148     4680.0     4680.0     4680.0     0     0     1     -360     360;
    39     46     0.000533240997229917     0.007146514113826056     0.943129990711823     4680.0     4680.0     4680.0     0     0     1     -360     360;
    40     46     0.0003185595567867036     0.004269346093974008     0.5634283061395305     4680.0     4680.0     4680.0     0     0     1     -360     360;
    42     46     0.000554016620498615     0.007424949728650447     0.24496882875631765     2340.0     2340.0     2340.0     0     0     1     -360     360;
    43     48     0.0003947368421052632     0.005290276681663444     0.6981611619555053     4680.0     4680.0     4680.0     0     0     1     -360     360;
    46     48     0.0003739612188365651     0.005011841066839052     0.6614158376420577     4680.0     4680.0     4680.0     0     0     1     -360     360;
    47     48     0.0008310249307479224     0.011137424592975672     1.4698129725379059     4680.0     4680.0     4680.0     0     0     1     -360     360;
    47     49     0.0004155124653739612     0.005568712296487836     0.7349064862689529     4680.0     4680.0     4680.0     0     0     1     -360     360;
    48     50     0.00046398891966759005     0.006218395397744749     0.8206455763336642     4680.0     4680.0     4680.0     0     0     1     -360     360;
    49     50     0.0013157894736842107     0.017634255605544814     2.3272038731850175     4680.0     4680.0     4680.0     0     0     1     -360     360;
    34     51     0.0003808864265927978     0.005104652938447182     0.6736642790798736     4680.0     4680.0     4680.0     0     0     1     -360     360;
    1     2     0.000171398891966759     0.0006200511816295644     20.822350444287004     3633.0     3633.0     3633.0     0     0     1     -360     360;
    1     4     0.00034626038781163435     0.004640593580406529     0.6124220718907941     4680.0     4680.0     4680.0     0     0     1     -360     360;
    3     5     0.0002839335180055402     0.0038052867359333544     0.5021860989504512     4680.0     4680.0     4680.0     0     0     1     -360     360;
    4     5     0.00018005540166204988     0.0024131086618113954     0.31845947738321295     4680.0     4680.0     4680.0     0     0     1     -360     360;
    5     6     0.0003116343490304709     0.0041765342223658766     0.5511798647017148     4680.0     4680.0     4680.0     0     0     1     -360     360;
    6     7     0.00023545706371191138     0.0031556036346764398     0.41644700888574004     4680.0     4680.0     4680.0     0     0     1     -360     360;
    6     8     9.695290858725763e-05     0.0012993662025138283     0.17147818012942237     4680.0     4680.0     4680.0     0     0     1     -360     360;
    8     57     0.00047783933518005545     0.0064040191409610115     0.845142459209296     4680.0     4680.0     4680.0     0     0     1     -360     360;
    52     53     0.0002493074792243767     0.003341227377892701     0.4409438917613718     4680.0     4680.0     4680.0     0     0     1     -360     360;
    52     56     0.00034626038781163435     0.004640593580406529     0.6124220718907941     4680.0     4680.0     4680.0     0     0     1     -360     360;
    53     58     0.00036703601108033245     0.004919029195230921     0.6491673962042418     4680.0     4680.0     4680.0     0     0     1     -360     360;
    53     54     9.695290858725763e-05     0.0012993662025138283     0.17147818012942237     4680.0     4680.0     4680.0     0     0     1     -360     360;
    54     55     0.0001038781163434903     0.001392178074121959     0.18372662156723824     4680.0     4680.0     4680.0     0     0     1     -360     360;
    55     56     5.54016620498615e-05     0.0007424949728650447     0.09798753150252706     4680.0     4680.0     4680.0     0     0     1     -360     360;
    56     57     9.002770083102494e-05     0.0012065543309056977     0.15922973869160648     4680.0     4680.0     4680.0     0     0     1     -360     360;
    54     59     0.0003808864265927978     0.005104652938447182     0.6736642790798736     4680.0     4680.0     4680.0     0     0     1     -360     360;
    58     59     0.00019390581717451525     0.0025987324050276567     0.34295636025884474     4680.0     4680.0     4680.0     0     0     1     -360     360;
    58     60     0.0002700831024930748     0.003619662992717093     0.47768921607481946     4680.0     4680.0     4680.0     0     0     1     -360     360;
    59     61     0.0006371191135734072     0.008538692187948015     1.126856612279061     4680.0     4680.0     4680.0     0     0     1     -360     360;
    61     62     0.0006786703601108033     0.009095563417596798     1.2003472609059564     4680.0     4680.0     4680.0     0     0     1     -360     360;
    62     63     0.00017313019390581717     0.0023202967902032647     0.30621103594539706     4680.0     4680.0     4680.0     0     0     1     -360     360;
    62     9     0.0002077562326869806     0.002784356148243918     0.36745324313447647     4680.0     4680.0     4680.0     0     0     1     -360     360;
    9     10     0.00034626038781163435     0.004640593580406529     0.6124220718907941     4680.0     4680.0     4680.0     0     0     1     -360     360;
    10     11     0.00023545706371191138     0.0031556036346764398     0.41644700888574004     4680.0     4680.0     4680.0     0     0     1     -360     360;
    11     12     0.0005886426592797783     0.007889009086691101     1.04111752221435     4680.0     4680.0     4680.0     0     0     1     -360     360;
    12     14     0.0004293628808864266     0.0057543360397040964     0.7594033691445847     4680.0     4680.0     4680.0     0     0     1     -360     360;
    12     13     0.00011542012927054478     0.0015468645268021765     0.4593165539180956     7019.0     7019.0     7019.0     0     0     1     -360     360;
    13     14     0.00019390581717451525     0.0025987324050276567     0.34295636025884474     4680.0     4680.0     4680.0     0     0     1     -360     360;
    14     16     0.0004362880886426593     0.0058471479113122275     0.7716518105824006     4680.0     4680.0     4680.0     0     0     1     -360     360;
    15     16     9.002770083102494e-05     0.0012065543309056977     0.15922973869160648     4680.0     4680.0     4680.0     0     0     1     -360     360;
    15     17     0.0003116343490304709     0.0041765342223658766     0.5511798647017148     4680.0     4680.0     4680.0     0     0     1     -360     360;
    14     17     0.00046398891966759005     0.006218395397744749     0.8206455763336642     4680.0     4680.0     4680.0     0     0     1     -360     360;
    13     15     0.00024238227146814408     0.003248415506284571     0.4286954503235559     4680.0     4680.0     4680.0     0     0     1     -360     360;
    16     19     0.0011357340720221608     0.015221146943733417     2.0087443958018047     4680.0     4680.0     4680.0     0     0     1     -360     360;
    17     18     0.0003289473684210527     0.004408563901386203     2.3272038731850175     9359.0     9359.0     9359.0     0     0     1     -360     360;
    18     20     0.00034626038781163435     0.004640593580406529     0.6124220718907941     4680.0     4680.0     4680.0     0     0     1     -360     360;
    18     21     0.0008310249307479224     0.011137424592975672     0.36745324313447647     2340.0     2340.0     2340.0     0     0     1     -360     360;
    19     22     0.00011772853185595569     0.0015778018173382199     0.20822350444287002     4680.0     4680.0     4680.0     0     0     1     -360     360;
    19     20     9.695290858725763e-05     0.0012993662025138283     0.17147818012942237     4680.0     4680.0     4680.0     0     0     1     -360     360;
    20     21     0.00019390581717451525     0.0025987324050276567     0.34295636025884474     4680.0     4680.0     4680.0     0     0     1     -360     360;
    22     23     0.00033240997229916895     0.004454969837190268     0.5879251890151623     4680.0     4680.0     4680.0     0     0     1     -360     360;
    20     24     0.0007825484764542938     0.010487741491718756     1.3840738824731946     4680.0     4680.0     4680.0     0     0     1     -360     360;
    23     24     0.0013434903047091413     0.018005503091977335     0.5940494097340703     2340.0     2340.0     2340.0     0     0     1     -360     360;
    64     65     0.00110803324099723     0.014849899457300895     0.4899376575126353     2340.0     2340.0     2340.0     0     0     1     -360     360;
    64     66     0.00029085872576177285     0.0038980986075414846     0.5144345403882671     4680.0     4680.0     4680.0     0     0     1     -360     360;
    65     70     0.002063711911357341     0.027657937739222916     0.9125088871172833     2340.0     2340.0     2340.0     0     0     1     -360     360;
    65     67     0.0008725761772853186     0.011694295822624455     0.3858259052912003     2340.0     2340.0     2340.0     0     0     1     -360     360;
    66     69     0.0007063711911357341     0.009466810904029321     0.312335256664305     2340.0     2340.0     2340.0     0     0     1     -360     360;
    66     67     0.0017313019390581717     0.023202967902032648     0.7655275898634927     2340.0     2340.0     2340.0     0     0     1     -360     360;
    67     71     0.0016481994459833795     0.022089225442735082     0.728782265550045     2340.0     2340.0     2340.0     0     0     1     -360     360;
    68     69     0.0009418282548476455     0.012622414538705759     0.41644700888574004     2340.0     2340.0     2340.0     0     0     1     -360     360;
    69     72     0.0007617728531855956     0.010209305876894365     0.3368321395399368     2340.0     2340.0     2340.0     0     0     1     -360     360;
    72     70     0.000554016620498615     0.007424949728650447     0.24496882875631765     2340.0     2340.0     2340.0     0     0     1     -360     360;
    70     71     0.0018698060941828255     0.025059205334195263     0.826769797052572     2340.0     2340.0     2340.0     0     0     1     -360     360;
];

%% DC grid topology
mpc.dcpol = 2;%% DC bus data
%column_names%   busdc_i    grid    Pdc    Vdc    basekVdc    Vdcmax    Vdcmin    Cdc
mpc.busdc = [
    1     2     0.0     1.0     320     1.05     0.95     0;
    2     2     0.0     1.0     320     1.05     0.95     0;
    3     1     0.0     1.0     320     1.05     0.95     0;
    4     1     0.0     1.01     320     1.05     0.95     0;
    5     3     0.0     1.0     525     1.05     0.95     0;
    6     3     0.0     1.01     525     1.05     0.95     0;
    7     6     0.0     1.01     525     1.05     0.95     0;
    8     6     0.0     1.0     525     1.05     0.95     0;
    9     4     0.0     1.01     525     1.05     0.95     0;
    10     4     0.0     1.0     525     1.05     0.95     0;
    11     5     0.0     1.01     525     1.05     0.95     0;
    12     5     0.0     1.0     525     1.05     0.95     0;
];

%% DC branch data
%column_names%    fbusdc    tbusdc    r    l    c    rateA    rateB    rateC    status
mpc.branchdc = [
    3     4     0.00017008463541666666     0     0     4700.0     4700.0     4700.0     1;
    1     2     0.0004020182291666666     0     0     2200.0     2200.0     2200.0     1;
    5     6     0.00015510204081632654     0     0     1400.0     1400.0     1400.0     1;
    9     10     0.00037913832199546485     0     0     4000.0     4000.0     4000.0     1;
    11     12     0.00022058956916099772     0     0     4000.0     4000.0     4000.0     1;
    7     8     0.0002757369614512472     0     0     2000.0     2000.0     2000.0     1;
];

%% AC/DC converter data
%column_names%    busdc_i    busac_i    type_dc    type_ac    P_g    Q_g    islcc    Vtar    rtf    xtf    transformer    tm    bf    filter    rc    xc    reactor    basekVac    Vmmax    Vmmin    Imax    status    LossA    LossB    LossCrec    LossCinv    droop    Pdcset    Vdcset    dVdcset    Pacmax    Pacmin    Qacmax    Qacmin
mpc.convdc = [
    1     76     1     1     0.0     0.0     0     1     0.0     0.0     0     1     0.0     0     0.0004163223140495868     0.0053549874777098746     0     220     1.2     0.85     1.1     1     2.206     0.887     1.4425     1.4425     0     0.0     1.0     0     1200     -1200     1200     -1200;
    2     16     2     2     0.0     0.0     0     1     0.0     0.0     0     1     0.0     0     0.0004163223140495868     0.0053549874777098746     0     220     1.2     0.85     1.1     1     2.206     0.887     1.4425     1.4425     0     0.0     1.0     0     1200     -1200     1200     -1200;
    3     75     1     1     0.0     0.0     0     1     0.0     0.0     0     1     0.0     0     0.0002081611570247934     0.0026774937388549373     0     220     1.2     0.85     1.1     1     4.412     0.887     0.72125     0.72125     0     0.0     1.0     0     1200     -1200     1200     -1200;
    4     9     2     2     0.0     0.0     0     1     0.0     0.0     0     1     0.0     0     0.0002081611570247934     0.0026774937388549373     0     220     1.2     0.85     1.1     1     4.412     0.887     0.72125     0.72125     0     0.0     1.01     0     1200     -1200     1200     -1200;
    5     2     2     2     0.0     -0.85     0     1     0.0     0.0     0     1     0.0     0     0.000625     0.007951345042660117     0     220     1.2     0.85     1.1     1     2.206     0.887     1.4425     1.4425     0     -0.85     1.0     0     800     -800     800     -800;
    6     1     1     2     0.0     -0.99     0     1     0.0     0.0     0     1     0.0     0     0.000625     0.007951345042660117     0     220     1.2     0.85     1.1     1     2.206     0.887     1.4425     1.4425     0     -0.99     1.01     0     800     -800     800     -800;
    7     74     1     1     0.0     19.87     0     1     0.0     0.0     0     1     0.0     0     0.0004163223140495868     0.0053549874777098746     0     220     1.2     0.85     1.1     1     2.206     0.887     1.4425     1.4425     0     19.87     1.01     0     1000     -1000     1000     -1000;
    8     60     2     2     0.0     0.0     0     1     0.0     0.0     0     1     0.0     0     0.0002081611570247934     0.0026774937388549373     0     220     1.2     0.85     1.1     1     4.412     0.887     0.72125     0.72125     0     0.0     1.0     0     1000     -1000     1000     -1000;
    9     77     1     1     0.0     19.87     0     1     0.0     0.0     0     1     0.0     0     0.0002081611570247934     0.0026774937388549373     0     220     1.2     0.85     1.1     1     4.412     0.887     0.72125     0.72125     0     19.87     1.01     0     1000     -1000     1000     -1000;
    10     10     2     2     0.0     0.0     0     1     0.0     0.0     0     1     0.0     0     0.0002081611570247934     0.0026774937388549373     0     220     1.2     0.85     1.1     1     4.412     0.887     0.72125     0.72125     0     0.0     1.0     0     1000     -1000     1000     -1000;
    11     73     1     1     0.0     29.63     0     1     0.0     0.0     0     1     0.0     0     0.0003125     0.0039756725213300585     0     220     1.2     0.85     1.1     1     4.412     0.887     0.72125     0.72125     0     29.63     1.01     0     800     -800     800     -800;
    12     23     2     2     0.0     0.0     0     1     0.0     0.0     0     1     0.0     0     0.0002081611570247934     0.0026774937388549373     0     220     1.2     0.85     1.1     1     4.412     0.887     0.72125     0.72125     0     0.0     1.0     0     1000     -1000     1000     -1000;
];

%% Generator data
%    bus    Pg    Qg    Qmax    Qmin    Vg    mBase    status    Pmax    Pmin    Pc1    Pc2    Qc1min    Qc1max    Qc2min    Qc2max    ramp_agc    ramp_10    ramp_30    ramp_q    apf
mpc.gen = [
    4     0     0     786.0     -786.0     1.0     100     1     1572.0     0     0     0     0     0     0     0     0     0     0     0     0;
    7     0     0     786.0     -786.0     1.01     100     1     1572.0     0     0     0     0     0     0     0     0     0     0     0     0;
    9     0     0     2335.0     -2335.0     1.01     100     1     4670.0     0     0     0     0     0     0     0     0     0     0     0     0;
    10     0     0     2335.0     -2335.0     1.01     100     1     4670.0     0     0     0     0     0     0     0     0     0     0     0     0;
    11     0     0     2335.0     -2335.0     1.01     100     1     4670.0     0     0     0     0     0     0     0     0     0     0     0     0;
    13     0     0     2335.0     -2335.0     1.01     100     1     4670.0     0     0     0     0     0     0     0     0     0     0     0     0;
    21     0     0     872.0000000000001     -872.0000000000001     1.01     100     1     1744.0000000000002     0     0     0     0     0     0     0     0     0     0     0     0;
    24     0     0     1165.0     -1165.0     1.01     100     1     2330.0     0     0     0     0     0     0     0     0     0     0     0     0;
    25     0     0     894.5     -894.5     1.0     100     1     4670.0     0     0     0     0     0     0     0     0     0     0     0     0;
    45     0     0     894.5     -894.5     1.01     100     1     4670.0     0     0     0     0     0     0     0     0     0     0     0     0;
    47     0     0     894.5     -894.5     1.01     100     1     4670.0     0     0     0     0     0     0     0     0     0     0     0     0;
    49     0     0     894.5     -894.5     1.01     100     1     4670.0     0     0     0     0     0     0     0     0     0     0     0     0;
    57     0     0     1506.5     -1506.5     1.01     100     1     3013.0     0     0     0     0     0     0     0     0     0     0     0     0;
    61     0     0     1506.5     -1506.5     1.01     100     1     3013.0     0     0     0     0     0     0     0     0     0     0     0     0;
    71     0     0     1086.5     -1086.5     1.0     100     1     2173.0     0     0     0     0     0     0     0     0     0     0     0     0;
    72     0     0     1086.5     -1086.5     1.01     100     1     2173.0     0     0     0     0     0     0     0     0     0     0     0     0;
    44     0.0     0.0     545.0     -545.0     1.01     100     1     545.0     490.5     0     0     0     0     0     0     0     0     0     0     0;
    2     14.382328554999999     0     0     0     1.01     100     1     1438.2328555     0.0     0     0     0     0     0     0     0     0     0     0     0;
    33     7.079410104     0     0     0     1.01     100     1     707.9410104     0.0     0     0     0     0     0     0     0     0     0     0     0;
    73     12.10678371     0     0     0     1.0     100     1     1210.678371     0.0     0     0     0     0     0     0     0     0     0     0     0;
    74     7.88588516     0     0     0     1.0     100     1     788.588516     0.0     0     0     0     0     0     0     0     0     0     0     0;
    75     16.588626300719998     0     0     0     1.0     100     1     1658.8626300719998     0.0     0     0     0     0     0     0     0     0     0     0     0;
    76     7.4923184946     0     0     0     1.0     100     1     749.23184946     0.0     0     0     0     0     0     0     0     0     0     0     0;
    77     14.07008168     0     0     0     1.0     100     1     1407.0081679999998     0.0     0     0     0     0     0     0     0     0     0     0     0;
];

%% Generator cost data
%    2    startup    shutdown    n     c(n-1)    c(n-2)    c0
mpc.gencost = [
    2     0     0     3     0     97.27     0;
    2     0     0     3     0     97.27     0;
    2     0     0     3     0     95.67     0;
    2     0     0     3     0     95.67     0;
    2     0     0     3     0     95.67     0;
    2     0     0     3     0     95.67     0;
    2     0     0     3     0     81.26     0;
    2     0     0     3     0     81.26     0;
    2     0     0     3     0     108.23     0;
    2     0     0     3     0     108.23     0;
    2     0     0     3     0     108.23     0;
    2     0     0     3     0     108.23     0;
    2     0     0     3     0     95.82     0;
    2     0     0     3     0     95.82     0;
    2     0     0     3     0     79.44     0;
    2     0     0     3     0     79.44     0;
    2     0     0     3     0     108.23     0;
    2     0     0     3     0     0     0;
    2     0     0     3     0     0     0;
    2     0     0     3     0     0     0;
    2     0     0     3     0     0     0;
    2     0     0     3     0     0     0;
    2     0     0     3     0     0     0;
    2     0     0     3     0     0     0;
];

%% Adds current ratings to branch matrix
%    c_rating_a
mpc.branch_currents = [
4680.0;
9359.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
2633.0;
1975.0;
9359.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
2340.0;
2340.0;
4680.0;
4680.0;
4680.0;
4680.0;
2340.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
3633.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
7019.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
9359.0;
4680.0;
2340.0;
4680.0;
4680.0;
4680.0;
4680.0;
4680.0;
2340.0;
2340.0;
4680.0;
2340.0;
2340.0;
2340.0;
2340.0;
2340.0;
2340.0;
2340.0;
2340.0;
2340.0;
];
