
是self attention 沒有softmax以後有什麼好處呢 那linear attention 這樣子的RNN 他就是influence的時候 你用他來產生東西的時候 他像是一個RNN 在訓練的時候 你就展開 把他當transformer來train 他就transformer少了softmax啊 他完全可以套用 transformer的平行化的方法 來直接加速訓練的過程 所以假設剛才的 講attention的時候 到底怎麼加速的 你其實沒有聽得很清楚的話 也沒有關係 你只要記得 self attention是一個可以加速的東西 長得像self attention一樣 類似的這個level架構 都可以用類似的方法來加速 而linear attention 根本就是self attention 拿掉softmax 所以self attention 能怎麼展開 能怎麼加速 linear attention 就能怎麼展開 能怎麼加速 訓練的時候 像是一個self attention 但是influence的時候 他就像是一個RNN 就是這麼神奇
              
                  46:17
                  好 那也許講到這邊 你會覺得說 哎呀這個linear attention 這邊有一個奇奇怪怪的式子 VT乘以KT的transpose 感覺沒有非常的直觀 那我現在提供給你一個直觀的解釋 來解釋linear attention 為什麼這樣運作 好 那linear attention 為什麼這樣運作呢 那在解釋之前呢 我們先定義一下這整個式子的dimension 那在做linear attention的時候 你需要先算出一個V 算出一個K 算出一個Q 我們假設Q是一個 D為的 呃 向量 那它是由XT乘上一個矩陣得到的 K呢也是一個D為的向量 XT乘上一個矩陣得到的 V呢是一個D'為的向量 XT乘上一個矩陣得到的 D'可以等於D 在剛才說明裡面 我都是預設D'等於D 但D'其實是可以不等於D的 只要Q跟K的dimension一樣就好了 他們就可以做inner product
              
                  47:22
                  其實V沒有必要跟Q跟K的dimension一樣 V的dimension可以失誤 是不一樣的 好 那我們來看這個 update那個memory的式子 HT等於HT-1 加上VT乘KT的transpose 把它畫出來的話 就是HT等於HT-1 加上VT乘VT的transpose VT乘VT的transpose 是一個矩陣 那我們等於就是把這個資訊 加到memory裡面 VT乘KT的transpose到底是什麼意思呢 我們來想想看 VT乘KT的transpose 到底是什麼 這個矩陣啊 他的每一個column都是 VT的倍數 這個矩陣綠色的矩陣 每一個column都是VT 前面乘上一個scalar 這個scalar的數值 由K裡面的數值所決定 第一個column的VT就乘上 K這個向量的 第一個dimension 第二個column的VT就乘上 K這個向量的第二個dimension
              
                  48:25
                  以此類推 這個矩陣加到 HT-1 我們會拿這個矩陣來更新 HT-1 這個VT是什麼意思 這個VT就是我們預計 要寫入memory 寫入hidden state H的資訊 而K指的是什麼 K指的是要把這個資訊 寫到哪裡去 舉例來說 假設K這個向量 只有第二維是1 其他維度都是0 我們把V這個資訊 寫到H這個memory H這個hidden state 它是一個metric 這個metric的第二個column 當然你可以寫到不只一個column 你可以把他們的資訊分散到 不同的column去 所以你知道V是要被寫進去的資訊 而K他決定了 這個資訊要被寫進哪裡 是不是覺得 這個式子其實也挺直觀的呢 好那 怎麼得到輸出 把H乘上QT
              
                  49:29
                  得到YT 那這件事情的含義是什麼呢 HT我們剛才說他每一個column 就是存了不同的資訊 由K來決定什麼資訊 要被存到哪裡 而Q就是把資訊 從column中取出來 Q決定要從哪一個column取多少資訊 假設現在這個Q是0100 只有第二維是1 那就是從第二維 從這個H的第二個column 把資訊取出來 這個就是linear attention 那其實linear attention 從來都不是 什麼新的想法 linear attention是 20年的時候就已經 知道的東西了 最早就我所知啊這個linear attention的paper 應該是transformers RNN這篇paper所提出來的 從他的標題就知道他想幹嘛 他就想告訴你說transformer 他發現原來就是RNN 他們兩個只差了一個softmax而已 就是這麼神奇 所以這是20年就已經知道的事情
              
                  50:33
                  然後在機器學習這門課的22年 我們有講過linear attention 只是那個時候是從另外一個角度來切入的 那個時候是從transformer一路講過來 說transformer怎麼簡化變成linear attention 那這一次是反過來講 說RNN怎麼變成linear attention 就像是一個transformer一樣 好 那既然 那這個linear attention是 是20年就已經有了 這個20年 那時候應該現代智人 還沒有出現吧 陸西還在東非的大草原上吧 那個時候人類已經就知道 linear attention就是RNN了 但為什麼現在 linear attention沒有被用的 到處都是呢 因為linear attention實際上 幹不贏transformer的 是幹不贏self attention的 那為什麼linear attention 贏不夠self attention 很多人的解釋是這個樣子 他說你看RNN RNN他的記憶就是很小 那我剛才講過RNN的記憶並不小
              
                  51:37
                  他的記憶有多大 是你自己決定的 你可以開一個很大的矩陣 讓RNN有很大的記憶 那也許你說 那我就不說他記憶太小 我說他記憶是有限的 畢竟這個H的大小你開多大 他終究是有個上限吧 所以RNN的記憶是有限的 那你看Transformer 他可以Attend到最開始的Token 要Attend多長就Attend多長 他有無限的記憶 那RNG有限 當然是沒有問題 也非常的直觀 怎麼說RNG有限呢 假設第一個時間點 我們的K1是10000 那我們就是把V1寫到第一個Color 那K2是0100 就寫到第二個 就把V2寫到第二個Color K3是0010000 那就把V3寫到第三個Color 他假設最多隻有第一個Column的話 那最多隻能夠存 第一個不同的V 然後他們彼此之間不會受到幹擾 再有更多的V 你就可能會重疊起來了 就會存在這個重疊的Memory裡面
              
                  52:41
                  你的Memory就放不下 那個你存入的資訊 就會彼此互相幹擾了 所以RNN他能夠儲存的記憶有限 是非常直觀的 但接下來我要告訴你 Transformer能夠存的東西 也是有限的 怎麼說Transformer能夠存的東西 也是有限的呢 假設現在Q跟K的Dimension 是用小D來表示 如果今天我們的Sequence不長 Sequence比如說T 現在是小於等於D的狀態 那沒有問題 你完全可以設計出一組Key 設計出一組K 讓你的Memory之間 彼此不會衝突 我可以說K1是100 K2是0100 K3是0010 那如果我今天要把 V2取出來的話 我要把第二個時間點的資訊取出來的話 我只要Q設成0100 那0100 只跟第二個位置的K 算出來Attention是1 其他位置算出來的Attention的數值都是0 你就可以把V2
              
                  53:43
                  原封不動的取出來 但是這是在T 小於等於D的狀態下 如果今天Sequence 非常長 長過 D了呢 那就設個4096吧 那今天我們都希望T可以產生 T可以處理非常長的輸入 比如說100萬個Token 你總是有機會遇到 T是大於D的狀態 如果T大於D的時候 會遇到什麼樣的問題呢 你會遇到說明明你的Q 跟K2 它是一模一樣的 你的Query跟第二個位置的Key是一模一樣的 那照理說跟第二個位置的Key計算Inner Product 算出來Attention是1 但是在整個漫長的Sequence中 存在著某一個Key 這個Key 它在T'的地方出現 它跟Qt做Inner Product的時候 它算出來的Inner Product 是大於0的 那為什麼會這樣呢 那可以回去想想看 線性代數裡面學過的東西
              
                  54:48
                  在低微的空間中 你最多隻能夠找到 第一個orthogonal的vector 你找不到低加1個 這個orthogonal的vector 所以當T很長的時候 你總是會出現一些key 這些key跟你的query 你沒有要retrieve這個地方的資訊 但是這個key就是跟你的query 做inner product以後大於0 那你今天retrieve出來的資訊 就是不同位置的weighted sum 你的模型的記憶就開始錯亂 所以今天attention它能夠存的資訊 也是有上限的 其實linear attention存的資訊會有上限 self attention存的資訊就會有上限 因為他們其實是一樣的東西啊 他們並不是不一樣的東西 他們真的就只差了一個softmax而已 所以如果linear attention贏不過transformer 那差就是差在少了softmax 所以看起來softmax算是一個簡單的運作 他其實蠻重要的 好那linear attention他最大的問題是什麼呢 linear attention他最大的問題就是
              
                  55:52
                  就是他的記憶永遠不會改變 你看每一次ht-1 你不會做任何處理 就會被放到ht 那每一個時間點 你都會輸入xt過fb的資訊 進入ht-1裡面 輸入以後就永遠進去了 他就再也不會被做出任何改變了 這樣會造成什麼問題呢 這樣會造成模型永遠不會遺忘 他輸進去的東西就永遠輸進去了 他永遠不會改變 那softmax為什麼可以 做到記憶的改變呢 明明存的東西應該是一樣的 為什麼softmax有機會 做到記憶的改變呢 以下是我的解釋 這個我們想像說 現在是一個比較短的sequence 然後呢只有三個位置 你算出來的attention的weight 是0.6、1跟0.4 那你過softmax 1這個位置對應到的 過完softmax的attention的weight 是0.
              
                  56:53
                  45 好,那我們假設呢 在過softmax之前 Q跟K做inner product 算出數值是1的話 就代表這是一件很重要的事情 好,今天假設這個sequence 變得比較長,一樣有一些位置 算出 算出attention的位 過softmax之前的attention的weight是1 它是重要的事情 但一件事情有多重要 取決於在同一個sequence裡面 有沒有更重要的事情 如果更重要的事情 本來一件重要的事情 它就變得不重要了 因為在做softmax的時候 整個sequence裡面所有的attention 是會共同被考慮的 所有的attention之間 是會互相影響的 本來一件很重要的事情 如果出現更重要的事情 它就變得不重要了 本來1過完softmax 在這個sequence裡面是0.
              
                  57:47
                  45 但這邊attention的weight算出來是1 但是出現更重要的事情 attention的位算出來是2 那1 他固然收max以後 就變成只有0.17而已 所以一個事情的重要性 是相對於其他事情而言的 就好像在那個 在那個暗影軍團裡面 本來覺得愛恩很強 但是後來又出現了一個尖牙 你就覺得尖牙比愛恩更強 後來就沒有人再叫愛恩出來 只會叫尖牙出來 後來又出現了貝爾 貝爾又比尖牙更強 所以後來都是貝爾出來了 就是這樣 沒有人知道我在說什麼 我看了這個 我獨自升級以後啊 我有一個巨大的發現 我發現啊 原來千何以入侵事件 在日本官方的記載 跟韓國官方的記載 是非常不一樣的 我現在才知道 原來這個梅路愛姆 他就是貝爾 我現在才知道 原來濟州島 就叫做東果陀島 紅的事件 日千何以入侵事件 在兩個國家
              
                  58:50
                  兩個獵人協會官方的記載 是非常不一樣 那個我可以問一下 現在我獨自升級的動畫演到哪裡了嗎 有人可以告訴我嗎 演打蟻王了嗎 剛要打蟻王 那以下有爆雷啊 不想聽的人就把耳朵捂起來 但是我覺得那個接下來的劇情 你應該都猜得到了嘛 現在要去打 不想聽的人就把耳朵捂起來 現在要去打蟻王了對不對 後來呢 就是韓國獵人那邊 這個叫做 忘記他的名字了 反正一個姓崔的人 就發射了很多火球 然後同時間日本那邊呢 那個宙迪克 傑諾諾諾迪克呢 也發射了群星龍 就引開了螞蟻的注意 然後那些韓國獵人 就潛入了蟻穴 就看那個蟻後 然後就把蟻後殺了 然後蟻後被殺了以後 過一陣子蟻王才出現 就把那些韓國獵人通通都打趴下去 為什麼以後被攻擊的時候 以後明明被攻擊了很久 以後有八戶位
              
                  59:52
                  八戶位團 滅的時候以王都沒有出現呢 因為以王在另外一邊 被尼特羅纏住了 尼特羅不是跟他打了半天 然後他殺了尼特羅以後 尼特羅就放了一個薔薇 把他炸的手腳都斷了 然後他就吸收了其他護衛的能量 結果又復原了 然後這個時候他就發現了那韓國獵人 他復原了以後雖然還是很弱 因為那韓國獵人太弱了 所以他就回來把那韓國獵人通通都打趴了 然後後來這個我獨自升級的主角 叫什麼名字啊 叫陳正宇 陳正宇就出現 然後又把梅路艾姆幹趴 然後把他改了一個名字叫做貝爾 就這樣子 這解釋了很多事 你想想看 如果看日本官方那邊的記載 梅路艾姆被核彈炸了以後 他其實沒死 還復原了 但跟小麥下棋 嚇著嚇著不知道怎麼回事就死掉了 日本官方的解釋是核輻射死了 但以往這麼強 被炸斷四個手腳都會復活 他怎麼可能因為核輻射就死掉了 原來是被陳正宇殺了啊 這個故事我終於得到了滿意的答案 可惜可喝 就這樣子
              
                  1:00:55
                  總之就是這麼一個故事 所以我們今天呢 就學到說千合一個事件呢 在日本跟韓國記載是不一樣的 好那我們既然說 這個linear attention的問題 就是記憶不會改變 那我們能不能夠 就讓他的記憶會改變呢 比如說加上reflection的機制 剛才為了平行化 把reflection的機制丟掉了 把reflection的機制加進來 讓今天這個linear attention 可以逐漸遺忘 所以有一個network叫做 retention network red net 他做的事情就是在 ht-1的前面乘上一個 常數項叫做gamma 那這個gamma的作用就非常的直觀 他就是讓一些過去的記憶 逐漸被淡忘 但這個gamma你要設小於1 通常是設0到1之間 你讓gamma是0到1之間 那你就可以讓記憶逐漸被淡忘 好 在inference的時候 就是ht-1到ht中間要乘上 gamma那training的時候呢
              
                  1:01:58
                  對training而言 這邊我們實際上就不推到 你可以自己回去想看看 對training而言沒有什麼不同 唯一的不同只有算attention weight的時候 每一個attention weight 後面要再乘一個 gamma的次方向 alpha t attend到1的attention weight 要乘上gamma t-1 alpha t attend到i的attention weight 要乘gamma t-i 且描成一個gamma的四方向 它並不會影響平行化的過程 所以加一個gamma 沒問題 但是加一個gamma顯然是不夠的 每件事情都只會被淡忘 我們其實希望模型 有些事可以牢記 有些事可以淡忘 應該是依據不同的情境 決定要牢記還是要淡忘 所以後來就有一個RedNet更進階的版本 叫做Gated Retention Gated Retention 就是把Gamma改成Gamma T 也就是讓Gamma 可以隨著時間改變 那怎麼決定Gamma T的數值呢 那你就把XT
              
                  1:03:02
                  乘上一個Transform W Gamma再過一個Sigmoid 讓它輸出是0到1之間 就得到Gamma T 這個W Gamma它也是Network的參數 也是需要被訓練出來的 好,那所以呢 這個Gated Retention 就是在HT-1到 HT中間乘上一個Gamma T 這Gamma T是學出來的 每次都不一樣 讓模型決定什麼東西要被記得 什麼東西要被遺忘 也許看到一個換行的符號 代表進入一個新的段落 他就決定要遺忘 Gamma T就趨近於0 就可以把過去的資訊清空 那實際上在training的時候 你要把這整個運作的過程 展開回self-attention的樣子 那怎麼展開回self-attention的樣子呢 就是在本來的QK KV之外 你都要多計算一個γ 那如果是αt-1的話 就在後面乘上γi加1 γi加2 一直到γt 所以你就是改變了原來
              
                  1:04:06
                  attention weight的計算 那也不會影響 你最終訓練的時候 平行化的過程 但是這一些 reflection的方法 實在都太簡單了 有沒有更複雜的方法呢 有人就想說 我們設計一個矩陣叫做 GT 這個GT會跟HT-1 做element wise的相乘 這個符號代表element wise的相乘 如果我們可以把HT-1 跟GT做element wise的相乘 我們就可以操控 HT-1裡面每一個memory 要被記得還是 要被遺忘 但是如果這個GT 沒有做什麼特別設計的話 你其實就沒有辦法展開回原來 attention的樣子了 的時候加速就會有問題 那這邊直接告訴大家 結論如果GT 可以寫成ET乘上ST 的transpose一樣 可以把這樣子的reflection 寫回 self-attention的樣子 那後來在實作上
              
                  1:05:11
                  有人比過了 發現說這個ET呢 直接設成一個都是1的向量就好 就不需要再多決定 ET是什麼 這樣的結果跟ET 是一個學出來的數值 比起來是不會有什麼太大的差距的 這個1的意思就是一個向量 裡面所有的數值都是1 好 那到底 乘上這個GT 把這個H對GT 做element wise的相乘 代表什麼樣的含義呢 它的物理意義是什麼呢 我們來看一下這個GT 它是1乘上ST的transpose 它到底是什麼樣子 那這個GT它每一排 每一排每一個row 就是這個ST 它每一個column的數值都是一樣的 但是同一個column裡面數值的大小 取決於ST裡面數值的大小 當我們把這樣子的一個矩陣 跟H做element wise的相乘的時候 它代表了什麼樣的意思呢 假設在第一個dimension ST的第一個dimension是0
              
                  1:06:14
                  那相乘之後 就代表抹去這個column 原來的記憶 如果是1就代表記住這個column 原來的資訊 那如果是一個介於0到1之間的值 比如說0.1就代表 減弱這個column原來的記憶 所以看到GT呢 它就是對HT裡面 每一個column存的information 它決定要保留還是要減弱 還是要抹去 所以這個GT如果你把它寫成 1乘上ST的話 它是蠻有物理意義的 那事實上類似的設計 非常非常多 只能告訴你 汗牛充棟 那這是引用至某一篇paper裡面的table 他就告訴你說 有這麼多這麼多的network架構 他們通通都可以看作是廣義的recurrent neural network 那這個表格裡面用的符號跟我剛才用的符號略略有不同啦 他把hidden state用s來表示 那我是用h來表示 那你可以看到說 這一些network架構的設計
              
                  1:07:19
                  通通都是hidden state前面做點什麼事 再加上點什麼 得到新的hidden state 新的hidden state再做點什麼 就得到最終的輸出 他用O代表最終的輸出 好在這裡面 我們終於赫然發現了 Mamba 發現Mamba 講了這麼久 終於出現了Mamba 所以Mamba是什麼呢 你看Mamba 他是對ST-1 做了一個非常複雜的 element wise的相乘 至於為什麼會出現這麼複雜的東西 我們今天就不解釋 那你去看 原始的論文你會發現蠻難讀懂的 因為它是從 它是從continuous state space的概念開始講起 跟一般類神經網路切入的點呢 其實不太一樣 所以你不太清楚為什麼 它這樣子設計 不過這是第一代的Mamba 而且它這樣子設計之後 就會導致平行化有點問題 所以它必須要用一個叫做Scan的algorithm 來加速Mamba的訓練 它沒有辦法像self-attention一樣 展開成只有純粹的矩陣相乘 那後來到了Mamba2
              
                  1:08:21
                  你就會發現 Mamba2其實跟Gated retention是一樣的 它就把原來很複雜的設計 改成一個Time Dependent的Gamma而已 那在這一系列的類似的研究中 當然Mamba是最有名的 大家通通都聽過Mamba 我想它最有名的原因就是因為 它是真的能打的 而且它出來的又早 Mamba1是在23年的年底出現的 所以是遠古時代就已經有的一個設計 而在那個時代 確實沒有人打得贏Transformer 那個時候只有Mamba Mamba在這篇paper裡面大力強調 他是第一個真的能贏過Transformer的 這種Linear Attention的設計 所以這個圖上 橫軸是不同大小的模型 所以訓練的Flux不太一樣 他這邊是做到了 最大的是一點多B的模型 那縱軸呢 是Propensity是 你就想成Propensity越小代表模型越好 那橙色的這條線是Transformer++ 就Transformer再加了一些改進
              
                  1:09:26
                  比較好的Transformer 現在比較常用的一個Transformer 然後呢 Mamba是紫色的這條線 然後Mamba呢 是可以在多數狀況下 都微幅贏過Transformer++的 然後這個Mamba的paper就大力炫耀這件事 說這是第一次 這一種linear attention的架構 可以贏過 真正贏過Transformer 那Mamba當然 這些類似linear attention架構的設計 它最大的目的就是在influence的時候要加速 那Mamba當然提供了非常強大的加速 那這個圖上這個藍色跟綠色代表的是 Mamba橙色跟紅色代表的是transformer 縱軸是每秒可以處理多少token在influence的時候 你可以發現綠色跟藍色的bar都遠高於橙色紅色的bar 可以比transformer在influence的時候有更好的加速 前年年底出現的時候引起了一波轟動 大家都覺得哇,這個transformer的挑戰者出現了
              
                  1:10:33
                  那時候常常做的一個梗圖就是有一個巨大的石人 是Mamba,然後其他下面貴的都是transformer 我現在才知道原來那個圖是出自我獨自升級的梗圖啊 那另外一個想跟大家分享的這個linear attention 的變形呢,叫做delta net delta net的式子啊,你如果乍看之下發現它寫的奇奇怪怪的 什麼ht-1是identity metric減掉β乘上kt乘上kt的transpose 加βk乘上bt乘上bt的transpose 後面這一項你現在已經不會有覺得奇怪了吧 但前面怎麼會有kt乘上kt的transpose呢 到底想要搞什麼,到底想要做什麼事情 好,那這邊我們來解釋一下 為什麼delta net是這樣設計 這個是最原始的linear attention 這個最原始的linear attention 然後呢,delta net 他想要做的設計是 我們現在把資訊放到 memory裡面
              
                  1:11:36
                  但是memory在同樣的位置 他也有存一些資訊啊 我們能不能夠把 前面的資訊 把它先去除掉 所以他就看說 KT到底要放資訊到 哪些Column 然後把本來存在H裡面 那些Column的資訊把它去除掉 那他是怎麼做的呢 他先去計算一下 本來KT要去放置的那些Column 裡面到底存了什麼樣的V 所以他就把HT-1 存上KT得到 另外一個向量 這邊用VO的來表示 VO的是什麼意思呢 VO的就是原來存在 HT-1裡面的 資訊 如果我今天要用KT把它寫入的話 原來存在HT-1裡面的資訊長什麼樣子 然後呢 我們現在先減掉VO的 先把本來要存進去的資訊 先減 把本來要存進去的位置的那些資訊 先減掉 然後再把新的資訊加進去 所以它就像是遺忘的過程
              
                  1:12:41
                  我們本來要把資訊放到 memory的某個地方 那先把memory清空才好放進新的資訊 我們前後都成一個beta 代表說我們沒有要清到非常空 沒有要清到零 就清一定程度就好了 那至於要清多少程度 這個beta有一個下標K 代表說它也是讓network自己決定的 然後呢 把VOD用HT-1乘上KT 把它直接帶進去 帶進去以後就結束了 你把這個式子稍微整理一下 你就可以得到上面這個式子 但是這邊真正的 想要跟大家分享的 是一個腦洞大開的觀點 這個delta net接下來神奇的地方是 這個式子 這兩個式子前面都有βK 後面都有KT的transpose 把它們提出來 所以delta net更新HT的式子 又可以寫成HT等於HT-1 減掉βK 乘上HT-1乘KT 減掉VT再乘上KT的transpose
              
                  1:13:47
                  這邊都是簡單的矩陣運算 你可能覺得沒有什麼神奇的 但有人腦門一開說 這是gradient descent 這個式子是一個gradient descent的式子 h就是一種神秘的參數 它其實是memory 但這邊我們把它當參數來看 等號左邊的ht是gradient descent update後的參數 等號右邊的ht-1 是update前的參數 βk 這邊應該寫k嗎 c應該好像應該寫t才對 不過沒有關係 這個不是很重要 這個β是一個learning rate 就是gradient descent的learning rate 而β後面乘的這一項 ht-1kt-vt乘上kt的transpose 就是gradient 再來問題是 它是誰的gradient呢 它是這一個loss function的規定 這個loss function 如果你把h當作參數 像gradient的話 正好就會是紅色底線的這一個式子
              
                  1:14:52
                  而右邊這個loss 它到底代表什麼含義呢 右邊這個loss是把kt乘上h 再減掉vt 取它的弄平方再乘二分之一 這二分之一沒有很重要 這只是微分的時候 要讓你結果比較好看而已 hkt減掉vt的弄平方 是什麼意思 它是希望h這個memory 我用kt當作是query 從裡面取 取資訊出來的時候 取出來的結果 要跟vt 越接近越好 所以今天 我們每一個time step 我們會update memory裡面的內容 那怎麼更新memory的內容呢 更新的方向 就是要 讓用kt取出的資訊 跟vt越接近越好 所以 但這件事情其實很直觀 因為我們現在 就是用kt跟vt 把資訊放到h裡面嘛 那我們當然希望 接下來 用kt可以把vt完整的抽取出來 然後這邊有一個formulation 告訴我們說這個h更新的方向 就是要讓用kt當作query
              
                  1:15:57
                  可以從h裡面把vt取出來 代表我們今天把這個vt放進去的時候 我們接下來是可以用同一個kt把它取出來的 那事實上原來的linear attention 這個你也可以看作gradient descent 只是看作gradient descent以後 它的lt是另外一個lt就是了 那用這個方法 就有人發明瞭 Titan 就是Titan 這個今年1月的時候 不是有個很紅的東西叫做Titan 然後它的Title告訴你說 這是Learning to Memorize At Testing Time 它標榜的就是 它有一個神奇的Memory 這個Memory相關的參數 是在模型一邊做Inference的時候 一邊會改變的 其實它用的就是DeltaNet的概念 把原來Memory存的數值 當作是 參數把Memory Update 做Reflection這件事情 看作是做Gradient Descent 這個就是Titan Learning to Memory at Test Time 大家有興趣的話再去研究這篇文章
              
                  1:17:02
                  這是今年年初的文章 好那像 現在像這種漫拔類型的Network 最早他們都只 勸了一些比較小的 模型所以不知道 它是不是能真的實用 但是現在已經有比較大的 漫拔類型的模型 比如說GEMBA 是背上漫拔的一個 Language Model 它比較大的版本有52 那今年年初 有一個模型叫minimax01 minimax01也是用了 linear attention 那它是500多 它是400多B的那種大模型 所以linear attention真的可以 用來訓練巨大的語言模型 那也可以達到跟現在 transformer一樣好的結果 那這些linear attention的概念 也不只是用在 文字上,它也用在 影像上,那最近呢 有一個影像生成的模型 叫做SENA,SENA它標榜 就是它非常的快 它模型非常的小 那其中它用到的一個概念
              
                  1:18:05
                  就是linear attention 它用了linear attention 在inference的時候加快SENA 這一個network 當然在做影像處理的時候 並不是加上linear attention 是好的 有一篇paper叫做 MambaOut 那從他的title就知道 他想要講什麼 這篇paper的標題是 Do we really need Mamba for Vision 那MambaOut這篇paper在開頭就玩了一個梗啦 他就說 Kobe Bryant說過 What can I say Mamba out 結束了 這什麼意思呢 就是Kobe Bryant在退休的時候 他講的最後一句話 就是Mamba out Mamba其實就是Kobe Brian的綽號 他的綽號就是Black Mamba 我在想說為什麼大家講Mamba的時候 都是放個蛇呢 其實感覺放Kobe Bryant其實也是可以的 好 總之這邊Paper就是告訴你 他就玩了個梗告訴你說 Mamba也不一定要用在影像上 他說過去有很多有用影像的 這個SSM指的就是類似Mamba LinearAttention那一種Network架構 現在有很多這種影像版本的Mamba 他們就故意把影像版本的Mamba裡面的Mamba拿掉
              
                  1:19:11
                  然後看看會不會比較好 但發現在分類的任務上 Mamba Out的模型表現 是比其他模型要好的 所以看起來Mamba在影像分類上並沒有幫助 但是其實你要注意啊 這篇文章並不是覺得Mamba不需要 因為你想看這篇文章啊 他是把類似這種Attention的架構都拿掉 他真正表明的是在分類的問題上 你不見得需要用到Attention這種架構 因為Attention他的目標就是考慮非常大的範圍 在分類的問題上可能不需要考慮這麼大的範圍 只要用CNN就足夠了 在同一篇paper裡面 他其實也有說 這個有Mamba的架構 在其他影像的任務上 比如說影像分割上面 還是會做的 比沒有這種self-attention的架構 或沒有Mamba的架構 做的還要好的 好 那今天另外一個流行的趨勢就是 因為現在啊 那些off-the-shelf的language model 比如說Lama都非常的巨大 所以如果你想要做這種linear attention 有關的研究
              
                  1:20:13
                  你自己設計network的architecture 你自己訓練 你是怎麼訓練都幹不贏Lama的啦 那所以怎麼辦呢 所以現在一個流行的趨勢就是 不要再從頭做起 如果你只是想要研究把self-attention改一個樣子會怎樣 那你何不從現有的語言模型 比如說Lama開始fine-tune呢 你就把它裡面的self-attention拿掉 直接換成Mamba 然後再fine-tune 看看Mamba有沒有辦法發揮作用 或甚至很多人會嘗試說 把self-attention裡面的一些參數 保留下來 直接加到新的設計裡面 將讓新的network需要train的參數越少越好 那看看你能不能有一些設計 能贏過傳統的語言模型 那這邊就引用了大量的論文 那你從標題裡面可以知道他們在做什麼 比如說the Mamba in the Lama 或transformer to SSM 他們就是從一個現有的 self-attention-based language model 開始做起 那微調中間跟self-attention有關的layer 那可以展示說插入一些attention attention的變形還是有用的
              
                  1:21:18
                  好那最後幾兩頁投影片呢 就是其實線上有一個賭局 這個賭局賭的是 is attention all you need 那這個賭局是說 attention是從2017年開始 10年之後到2027年 人們是不是還會覺得 attention是最強的network架構 那現在離驗證這個預言 差656天 然後呢 這個預言就是想說 到2027年1月1號的時候 到底self-attention transformer-based的架構 是不是還是霸榜的 支持這個選項的人 是Jonathan Frankle 他是這個哈佛的教授 也是Mosaic ML的 chief scientist 反對這個proposal的人 是Sasha Rush 那他是Cornell的教授,也是HuggingFace的research scientist,他們就是賭,應該是賭各自在公司裡面的股份這樣子,對,有這麼訝異嗎? 然後看看最後,2027年的時候,到底Transformer還會不會繼續霸榜
              
                  1:22:29
                  那Sasha Rush呢,其實他也是個YouTuber,那我其實從他的影片裡面學到很多東西 所以這邊把他的名字放在這邊,也推薦給大家另外一個講機器學習的YouTube
              
            