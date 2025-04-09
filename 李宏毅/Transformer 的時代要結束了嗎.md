
不同的 類神經網路的時候 你的loss長什麼樣子 那其實這篇paper最主要想要 分析的其實也不是skip connection 他就是發明了一個 visualization的方法
              
                  05:27
                  一個可視化的方法 可以讓你看到error的surface長什麼樣子 因為實際上error的surface 應該是存在一個高維的空間中 參數是非常高維嘛 他想辦法把那個 高維空間中的變化 投影到二維的平面 讓你可以想像一下 error的變化 loss的變化長什麼樣子 那這篇論文就分析了一下 沒有residual connection跟有residual connection 的loss的變化 發現說沒有residual connection loss非常的崎嶇 就像是一個山水化一樣 所以你比較難做optimization 比較容易卡在local minimum 或者是saddle point 而如果用residual connection的話 那error surface看起來比較平坦 所以optimization比較容易 所以這邊舉兩個例子是想要告訴你說 每一種架構 都有它存在的理由 而它們存在的理由不一定是一樣的 比如說CNN跟residual connection 它存在的理由是不一樣的 所以以後大家看到一個新的架構被提出來的時候 你要想想它是為了什麼而被設計出來的
              
                  06:35
                  那Transformer的出現是因為什麼樣的理由呢 那這邊要講得更精確一點 我們實際上指的是 在這門課裡面我們實際上討論的是 那Transformer是一個比較大的架構 那它其實有非常多不同的變形 事實上 2019年GPT2用的Transformer 跟現在LAMA用的Transformer 雖然都叫Transformer 中間還是有一些差別的 而這些差別其實會造成 他們顯著的能力的差異 那到時候在作業4 當你自己勸Transformer的時候 你會真的體驗一下這件事 你用LAMA的架構 跟用GPT2的架構 你會得到不一樣的結果 好 那我們這邊講的 並不是完整的Transformer 我們要討論的 而裡面的其中一個Layer 叫做Self-Attention Layer 好 那在討論Self-Attention Layer之前 我們要講的是Self-Attention Layer 是怎麼取代掉
              
                  07:39
                  在Self-Attention Layer之前 大家比較常用的RNN或LSTM 那你知道Self-Attention是怎麼取代RNN跟LSTM之後 你更能想像為什麼其他的架構 比如說Mamba會取代了Self-Attention 那這個Self-Attention在機器學習這門課呢 是從2019年開始引入的 後來從2021年開始課程裡面 就沒有出現RNN了 但事實上這一堂課呢 是會告訴你RNN又回來了 其實Mamba就是RNN 那等一下你會知道為什麼Mamba就是RNN 好 那這一些什麼RNN啊 Self-Attention啊或Mamba啊 他們真正想要處理的問題是什麼 他們要解決的問題類型是 輸入一個vector的sequence 要輸出另一個vector的sequence 輸入一個vector sequence以後 這一個layer要把輸入的資訊做某種混合 產生另外一個vector sequence
              
                  08:44
                  那如果在做語言模型的時候 通常每一個輸出都只能看左邊更之前的輸入 所以我們在輸出y1的時候 會只考慮x1的資訊 輸出y2的時候會只考慮x1 x2的資訊 輸出y3的時候會只考慮x1 x2 x3混合起來的資訊 以此類推 但在其他的應用裡面不一定是這樣 可以把整個input sequence合起來 混合起來再產生輸出的sequence 不過在等一下課程裡面 因為我們主要專注於討論語言模型 所以我們就假設說 每一個輸出的y都只能夠看到 比他更左邊的 在他之前就已經輸入的x3 好,那我們來看看 RNNN是怎麼解這個問題的 事實上RNN比較像是一整群不同架構的代稱 這邊在RNN後面加一個style 就是RNN流 RNN流的這種network架構 它的基本精神是 怎麼樣把輸入X1到XT混合起來
              
                  09:48
                  最後輸出YT呢 有一個東西叫做hidden state hidden state的作用 是把目前已經看到的輸入 全部混合起來存在hidden state裡面 輸出的時候 只需要根據hidden state 就可以決定現在的輸出 那講得更具體一點呢 RNN流的方法 它一個廣義的這個寫法呢 可以寫成這個樣子 這個h呢 代表的是hidden state裡面存的資訊 那我們這邊沒有特別告訴你 h它應該是一個vector 還是一個matrix 其實兩者都是可以的 所以h是存在hidden state裡面的資訊 那在第一個時間點 hidden state裡面的資訊是怎麼被得到的呢 它是由前一個時間點的hidden state的資訊 ht-1跟現在的輸入xt所共同決定的 ht-1通過一個fa 這個fa是一個函式
              
                  10:51
                  它長什麼樣子是你自己設計的 通常裡面有一些參數是需要透過訓練資料 訓練出來的 ht-1通過fa 再加上xt 通過fb得到ht ht再通過fc得到yt 那如果把它視覺化的話 整個運作的過程是這樣 首先有一個h0 現在輸入x1 根據x1跟h0 還有fa fb產生h1 h1通過fc產生y1 同樣步驟就反覆進行 輸入x1 x2 通過fb h1 加起來得到h2 h2再通過fc 再產生y2 那這過程就反覆繼續下去 輸入x3產生h3 再產生y3 一直到輸入xt產生ht 產生yt 那這個h0啊 我這邊沒有特別告訴你說 它是什麼樣子的東西 那最早年的RNN 通常H是一個向量
              
                  11:56
                  但其實它不一定要是向量 它也可以是一個很大的矩陣 那很多人對於RNN有一個誤解 覺得說RNN它hidden state 就是一個向量可以存的資訊很少 這是個誤解 它可以不要是向量 它可以是一個很大的矩陣 那更泛用的一個寫法呢 是會把這個FA、FB跟FC呢 加上一個下標 代表說呢 FA、FB、FC這三個函式 並不是一成不變的 它們是會隨著時間變化的 那怎麼讓FA、FB、FC隨著時間變化呢 那你可以讓這三個函式 跟X有關係 跟輸入的X有關係 你可以讓X來決定 FA、FB、FC長什麼樣子 那因為每一個時間點輸入的X都不一樣 那這三個函式 會隨著時間變化而不同 那也可以想見說 這樣子跟時間有關的操作
              
                  12:59
                  可以帶給這個RNN 很好的性質 比如說假設 今天對一個語言模型來說 他想要看完 一整個文章的段落以後 就遺忘過去的東西 然後有新的開始 那可以怎麼做呢 假設輸入的X2是一個 換行符號 那FA2就 執行清除這個動作 把之前存的資訊清掉 或者是說 假設X2是一個不重要的資訊 那乾脆FB2 可以採取關閉這樣的行為 不要讓X2的資訊 進入H2裡面去佔用 寶貴的儲存空間等等 所以把這個FAFBFC 讓他們跟輸入有關 你就可以做更複雜的操作 好 讓FABC跟輸入有關 這是什麼呢 其實這就是 LSTM 那我不知道還有多少人知道什麼是LSTM LSTM GRU這一些有
              
                  14:02
                  Gate的RNN 其實就是讓FA FBFC跟時間有關 也就是跟輸入有關 那這個圖看起來蠻複雜的 那就是取自過去投影片的圖 這個圖2021年以後 就再也沒有出現過了 這次是2021年以後 第一次再用這個圖 總之就是告訴你說 過去常用的RNN的變形 就是最簡單的RNN 以前叫Vanilla的RNN 最簡單的RNN 其實實用度很低 多數人都是用LSTM或GRU LSTM或GRU 就是讓FA、FB、FC 跟T有關 這個T就是一個Gate 那其實在LSTM裡面 他就是有三個Gate 一個叫做Input Gate 他就是對應到FBT 會決定模型 要不要忘記東西 就是對應到FAT 一個output gate決定什麼東西要被輸出來 就是FCT 那其實RNN流的做法 跟我們第二堂課講的
              
                  15:06
                  AI agent怎麼處理memory 也有一取同工之妙 我們在講AI agent的memory的時候 說我們要有三個模組 一個書寫的模組 決定什麼樣看到的東西要被放到記憶中 一個reflection的模組 對記憶中有的資訊 重新整理 一個讀取read的模組 從記憶中讀取資訊出來 而RNN流的這整套做法呢 H就是我們的記憶 FB就是書寫的模組 決定什麼東西要被放到記憶中 那FC就是讀取的模組 決定要怎麼從記憶中把資訊讀出來 而FA就是reflection反思的模組 決定記憶中的資訊要不要有 好 那以上呢就是跟大家介紹一個 RNN流最泛用的式子 好那在influence的時候 就假設你內部已經訓練好了 要使用它 比如說它是一個語言模型 你要用它來產生一個句子的時候
              
                  16:10
                  那RNN流的做法是怎麼運作的呢 首先你要有一個H0 那這邊要注意一下 這整個LLM 這整個語言模型 還可能有很多其他的layer 還可能有很多其他的部件 比如說MLP Fully Connected Feed Forward的部件 我們在這個圖上 只把RNN相關的部分畫出來 那RNN也不會只有一層 他也有很多層 所以叫想成說 當我畫這個圖的時候 我就是把原來大家常見的 Transformer裡面的Self-Attention Layer 換成RNN RNN層要有一個 輸出史的狀態叫H0 輸入第一個符號 通常第一個符號是Begin of Sentence 然後呢 產生X1 讓我們有H1 H1讓我們有Y1再送到 接下來的layer去處理 最後輸出一個字叫大 然後呢做autoregressive 大變成下一個時間的輸入 再產生X2再產生H2再產生Y2 再產生加
              
                  17:12
                  加再變成輸入 產生H3產生Y3 產生好同樣的步驟 就反覆反覆一直進行下去 這就是RNN 這時候所做的事情 好,這個是RNN 接下來我們來講 Self-Attention 我們來看看Self-Attention 相較於RNN 有什麼好處,憑什麼能夠取代 RNN 好,來看一下Self-Attention Self-Attention怎麼運作的呢? 輸入是一個Sequence X1到XT 然後現在輸入的每一個 Vector都乘上三個 不同的Transform,製造出 三個Vector,X1就製造 K1,Q1,X2 就製造出V2,K2,Q2 以此類推 那假設我現在要算 YT的時候,第T個位置 對應的輸出YT的時候 要怎麼計算呢? 我把第T個位置的輸入 產生出來的這個QT 去跟每一個位置的K 都算inner product 算個內積
              
                  18:16
                  那我用α來表示 這個內積後的數字 內積後得到的數字 這個α呢現在通常叫做 Attention的Weight 雖然它有Weight這個字 但它並不是一個參數 它並不是類神經網路的參數 但我們也叫它Attention的Weight 就是了 那這個αT1代表QT 跟K1的Inner Product 那αT2代表 QT跟K2的Inner Product 以此類推 那你會做一個Softmax 讓這些α的值合起來是零 那Softmax看起來是個不起眼的動作 等一下會告訴你 Softmax其實妙用無窮 然後這邊的α呢 跟每一個時間點的V呢 做相乘 最後就得到Y 這是SelfAttention 做的事情 那希望這個大家都已經很熟悉了 那在以下的課程裡面如果要簡化的話 我就會直接把X1到 XT直接指到YT 代表說YT對X1到XT 做Attention
              
                  19:19
                  然後做VT上 最後得到YT 那很多人誤以為Attention這個概念 是來自於Attention is all you need Google那篇2017的paper 其實不是Attention的概念 其實很早就有了 就我所知 最早可以回溯到Neural Turing Machine 還有Memory Network 他們是什麼時候的類神經網路呢 你看這邊的Archive連結 他們是2014年的類神經網路 那個時候地球上還沒有多細胞生物 只有單細胞生物的時候 就有Attention這樣子的概念了 而且這兩個提出Attention的內部架構 你仔細看看它的連結 都是在14年10月放上Archive 前後只差幾天而已 在這個只有單細胞生物的時代 就已經非常的捲了 那把Attention用到語言模型裡面 也從來都不是一個新的想法 比如說 我們實驗室在2016年的時候 就嘗試把Attention加到語言模型裡面
              
                  20:26
                  這個是劉大榮同學做的 他做的時候呢 還是大學生 現在都博士班畢業好多年了 所以這個是一個 整個寒武紀時代的東西 那個時候寒武紀時代放在 這個Language Model裡面的Attention 跟現在有什麼不同呢 就是沒有任何不同 就是你看 那個時候只是不叫做QKV而已 Notation不太一樣 產生一個這個叫做Query 不過當時是用K來表示它 然後呢 這個是Key 這個是Value 然後就做你所熟知的Attention 放在Language Model裡面 只是在Attention之前 還會先做LSTM 才做Attention 好 那我們來看 這個Attention的這個架構 在Inference的時候 在生成句子的時候 是怎麼生的 好 第一個輸入進來產生X1 那X1沒有更之前的東西 所以他只能自己跟自己做Attention 產生Y1產生第一個字大 好 那大呢 被作為第二個輸入產生X2 可以對第一看第二個位置
              
                  21:31
                  都做Attention產生Y2 然後產生加 然後以此類推 第三個時間點輸入的是加 可以對一二三個位置都做Attention 產生Y3產生好 這個步驟就持續繼續下去 持續繼續下去 每一次都要對前面所有的位置 做Attention 而來比較一下RNN跟Attention的這樣的架構 在Inference在生成的時候 有什麼樣的不同 如果是RNN每一步的運算量 都是固定的 如果是Attention的話 你越往後 箭頭就畫越多 越往後輸入的序列越長 運算量就越來越大 那如果你看Memory的話 對RNN來說 每一次我要產生新的輸出 出的時候 我只要記得 前一個時間點的H是什麼 我就能夠做運算 對Attention而言 我要產生Y6的時候 前面X1到X5發生了什麼事 通通要被記住 才能產生Y6 所以Attention也非常的耗費Memory
              
                  22:36
                  隨著Input的Sequence越長 Attention這個架構 需要的Memory 對Memory的需求就越大 那講到這邊有人會說 但Attention還是有好處的 你可以感覺說 這個RNN呢 他的Memory很小 只有這麼小塊 感覺應該沒辦法記很多資訊吧 Attention 感覺可以記無窮長的資訊 那等一下的課程會告訴你說 這是一個對RNN的誤解 等一下告訴你說 Attention可以存無限長的資訊 只是一個假象 所以講到這邊 你可能覺得 這個Attention就是一無是處 沒什麼好的地方 沒什麼好的地方 那為什麼從2017年到最近 Attention這麼dominate深度學習的領域呢 好 那我們來看看 當年Attention is all you need 這篇最知名的Attention相關的論文 它是怎麼說的 那你從它的title Attention is all you need 你其實可以發現 這篇文章它並不是發明了Attention 很多農場文以為
              
                  23:41
                  Attention是2017年這篇文章發明的 其實不是 這篇文章它的貢獻 不是發明Attention 而是拿掉Attention以外的東西 所以才叫做Attention is all you need 那時候人們覺得只有Attention 應該是沒有辦法好好做語言模型的 他們把Attention以外的東西拿掉 發現還是可以好好運作 大家就驚呆了 那Attention is all unique 這篇paper在他introduction裡面 他就說這個Attention的好處 也就是那時候他們就把Transformer跟Attention 混在一起當作同一個東西講 這個Transformer的好處 好處是什麼呢 他第一個最大的好處是 他在訓練的時候 可以更加平行化 當然他們也得到了不錯的結果 什麼叫訓練的時候 可以更加平行化呢 那在這堂課裡面 我們沒有花太多時間講 類神經網路的訓練 所以如果你想要深入瞭解 是怎麼被訓練出來的 除了看2021的課程以外 你可以更深入看過去的課程
              
                  24:45
                  過去有講過backpropagation 那是一個一小時左右的錄影 那還講了另外一個更深入的 怎麼自動做backpropagation的方法 也就是computational graph 那這個也是花了一個小時左右的時間講的 因為上課的時間是有限的 所以後來這部分就沒有放在實體的課堂上 但如果大家有興趣的話 還是鼓勵你把這些技術深入的去了解它 那其實瞭解訓練的原理 對你的幫助就是 更清楚為什麼有些network架構是這樣設計 Transformer它真正的設計的好處 並不是什麼無窮長的memory 那個就是個假象 那是個hallucination 它真正的好處是訓練的時候 可以更容易平行化 好 那我試著來解釋 為什麼它訓練的時候 可以更容易平行化 雖然我們沒有講 怎麼真的訓練network 但是訓練network基本的步驟 是這個樣子的 要做的事情就是要教一個 語言模型說
              
                  25:49
                  輸入Token Z1到ZT-1 然後呢你要產生 ZT 你要怎麼教語言模型這件事呢 第一個步驟是先算出 目前的答案 算出輸入Z1到ZT-1的時候 語言模型 目前的輸出是什麼 直到目前的輸出以後 再跟正確答案計算差異 計算完差異 你才能夠更新參數 所以更新參數之前 第一步是要能夠計算出 現有的答案 所以你可以想像說計算出現有的答案 這件事情做得越快 你就可以越有效的更新參數 而Transformer他厲害的地方就是 他可以快速的計算出 現有的答案 這件事是怎麼做的呢 假設我們今天 要教語言模型說一句話 大家好我是人工智慧 在概念上 你做的事情可能是這樣的 輸入代表起始的符號 然後告訴語言模型說 輸入代表起始的符號 那你要輸出大是正確答案
              
                  26:54
                  輸入代表起始的符號跟大 輸出加才是正確答案 輸入代表起始的符號 輸入大輸入加 是正確的答案 這個是概念上的做法 但是Transformer有一個 更有效的運算方法 這更有效的運算方法是什麼呢 我們今天教模型的時候 我們會有一個完整的句子 就是要教模型說出這個完整的句子 大家好我是人工智慧 實際上的操作是這個樣子的 你把目標 你把 要讓語言模型說出的這個句子 你把你的ground truth 通通向右移動一格 前面放起始的符號 塞給語言模型 語言模型 可以平行的 如果用的是transformer的話 transformer可以平行的 一次輸出每一個 有一個時間點 所有的答案 所有的答案跟正確答案的差異 你可以一次看這麼多的差異 去更新參數 所以transformer厲害的地方是
              
                  27:58
                  如果給他一個 完整的句子 他可以平行的 計算出在每一個 時間點語言模型 這個transformer 要輸出的下一個token是什麼 他可以一次平行計算出 begin of sentence 大後面會輸出哪一個token 大家後面會輸出哪一個token 大家好會輸出哪一個token 這一切是可以 透過transformer 透過self-attention的架構 被平行計算出來的 那如果講得更清楚一點的話 我們來看看當我們用 self-attention的layer的時候 如果我們一次給self-attention layer 一個完整的輸入 給他一整個sequence的輸入 不是一個一個token產生的 的情況下 這個self-attention會怎麼運作 你輸入一個完整的sequence sequence裡面的每一個token 都會變成一個向量 我們這邊用X1到X6來表示 X1到X6生成的步驟 可以是平行的
              
                  29:02
                  他們彼此之間並沒有任何關聯 所以你可以平行的 把X1到X6產生出來 接下來 做self-attention 你可以平行的產生Y1到Y6 因為Y1到Y6 它中間生成的過程 彼此之間沒有任何關聯性 所以Y1、Y2一直到Y6 這六個向量是可以平行 被生成出來的 平行生成出來Y1到Y6之後 你就可以平行的輸出 每一個時間點 這個語言模型要predict next token的時候 要猜下一個符號的時候 它會輸出的token是什麼 所以self-attention 它真正的好處是 給一個完整的輸入 可以同時產生 每一個時間點 語言模型預測下一個token的token 是什麼 給定一個完整的輸入 所以平行的輸出 每一個時間點 接下來要輸出的token是什麼 所以這個是transformer的好處 好 那另外一個觀點來看這個transformer
              
                  30:06
                  你可以發現說 這個self-attention的計算 是一個非常GPU-friendly的計算 是可以大幅應用GPU效能的計算 那有關這個GPU 在做語言模型的時候 扮演什麼樣的角色 下週呢 王秀軒助教 會有更詳盡的剖析 那在這堂課你就先記得說 只要你做的是矩陣運算 GPU就會覺得開心 他就會覺得 他被妥善利用了 好 那Transformer 其實就是一連串的矩陣運算 所以說Transformer是一連串的矩陣運算呢 現在綠色的這一排向量是輸入的X到 是輸入的X1到X6 好 那這每一個X呢 每一排X呢 成一個Transformation 變成每一個時間點的Query 每一個時間點的Key 每一個時間點的Value 接下來呢 我們要算Attention 要算Inner Product
              
                  31:10
                  那怎麼算Inner Product呢 把這個代表Key的矩陣做Transpose 直接乘上這個由Query所組成的矩陣 你就得到每一個時間點兩兩之間的Attention了 所以今天這個向量兩兩之間做Inner Product 其實可以看作是兩個矩陣直接相乘 不過有一些Attention我們是不要的 因為我們假設說只有後面的時間點 可以Attain到前面的時間點 所以有一些Attention我們就直接 我們就直接把它設為0 那些是我們就算計算出來 也不需要的 然後呢 你做一個SolveMax 得到另外一個Attention的Matrix 把這個Attention的Matrix 再乘上由ValueVector 所組成的矩陣 就得到輸出了 所以從輸入X到輸出X 通通都是矩陣運算 那假設這個步驟你沒有聽得很懂的話 反正你就記得 今天你只要弄成 Self-Attention那個樣子
              
                  32:13
                  那你的運算的過程 通通都是矩陣運算 這是一個可以讓GPU 歡喜的運算過程 可以有效的利用GPU的效能 好那另外一邊 我們就來看 為什麼RNN沒辦法 在訓練的時候 有效運用GPU的效能 今天假設在訓練的時候 一次給RNN完整的輸入 會發生什麼事呢 進入RNN這一層之前的X1到X6 你可以平行運算出來 但是RNN本身 是沒有辦法平行運算的 輸入X1之後 你才能夠產生H1 有了H1才能有H2 有了H2才能有H3456 所以你要計算H6 你必須把前面H1到H5 計算出來之後 你才能計算H6 那這是GPU最討厭的狀態 你就記得 討厭等待 那一些加速GPU的方法 其實就是避免GPU去等待
              
                  33:16
                  盡量不斷的塞事情給GPU做 像這種前面H5要算完才能算H6 這個是GPU最討厭的狀態 因為他必須要等前面的東西算完 他沒有辦法發揮他平行化的優勢 所以你如果要讓Y1到Y6一次產生出來 你得把H1到H6都算出來以後 你才能一次把Y1到Y6 而計算H1到H6這個步驟 是不容易平行化的 它是一個沒有辦法有效運算的步驟 所以現在我們可以比較 Self-Attention跟RNN流的做法 你就知道說 如果我們看Inference的時候 我們在使用這個語言模型 讓它產生一個序列的時候 如果Self-Attention 那你的計算量跟記憶體的需求 會隨著序列長度增加而增加 而 RNN計算量跟記憶體的需求是固定的 但是Self-Attention的好處是在訓練的時候 我們容易平行化
              
                  34:18
                  容易發揮GPU平行化的能力 而RNN它的壞處是 它難以平行化 所以在2017年到最近 人們選擇了Self-Attention 加快訓練的速度 但是RNN難以平行化後面 我們加了一個問號 那這堂課就是要告訴你 其實在訓練的時候 是可以平行化的 Self-Attention 在Inference的時候遇到長的sequence 就比較不利 但今天人們需要長的sequence 今天語言模型都需要處理非常長的輸入 比如說做IG的時候 語言模型從網路上搜尋了一大堆文章 作為它的輸入 你需要長的輸入 或AI agent要運算好幾個回合 跟環境不同 跟環境一直的互動 那也需要很長的sequence 今天又流行多模態的模型 那多模態的模型跟處理文字的模型 它背後的原理其實很像 你只是把語音或者是把影像 表示成token sequence而已
              
                  35:24
                  但是語音跟影像表示成token sequence 它會是比文字還要長的非常多的sequence 所以我們真的需要有效的處理長sequence的方法 那現在語言模型可以處理的sequence越來越長 然後以下這張圖片呢 是來自右下角這個連結 它就說在2022年 剛有GPT3.5的時候 它大概只能讀哈利波特的一個章節 但到GPT4的時候 它就可以讀完神秘的魔法石 一直讀到第二步 那如果是靠2.1 它就可以把哈利波特第一步跟第二步 神秘的魔法石跟消失的密室都讀完 如果是Gemini 1.5 它可以讀兩個Million的Token 它可以讀兩百萬個Token 它不只可以把哈利波特第一集 一直讀到最後一集 還可以幾乎把魔戒三部曲看完 所以現在這些語言模型 我們希望它可以讀很長的sequence 而Attention這個方法 在讀長sequence的時候是不利的 所以人們就開始懷念起RNN的好
              
                  36:30
                  所以有人就問了一個靈魂的叩問 RNN訓練的時候 真的 沒辦法平行嗎 我們來看看有沒有讓RNN平行的可能 好 那我們先把H1到HT通通都列出來 看看他們長什麼樣子 右邊是RNN最泛用的式子 那H1是H0過FA加上X1過FB H2是H1過FA加上X2過FB H3是H2過FA再加X3過FB 以此類推 那因為算H2的時候知道H1 算H3的時候知道H2 所以沒有辦法做平行運算 但真的是這樣嗎 我們能不能把這些式子展開來看看 我們就假設FAH0是一個zero matrix 所以我們就不考慮這一項 那H1等於FB of X1 H1等於X1通過FB
              
                  37:34
                  接下來呢 我們已經知道H1是什麼了 我們把H1直接塞進去 把H2展開 讓他的輸入沒有H1 把H1用X1通過FB替代掉 所以H2就變成X1通過FB再通過FA 加上X2通過FB 以此類推 把H2帶到H3的式子裡 讓H3的輸入沒有H2 所以就變成H3 就是X1通過FB 通過FA 再通過另外一個FA 然後這邊這個X2也要通過 這個有點複雜 這邊X2要通過FB 再通過FA 然後X3要通過FB 總之你可以把整個式子通通都展開 你可以把每一個H1到 你可以把H1到HT展開 讓他們彼此不要有dependent 但這樣真的就能夠平行運算了嗎
              
                  38:40
                  看起來是不行的 因為展開到最後 你會發現你的式子裡面 有非常長的連續的 函式的呼叫 比如說到HT的時候 X1要先通過FB 要再通過FA 再通過FA 再通過FAT-1 再通過FAT 要通過一連串的FA 才能夠計算出答案 這個 也是太dependent的 這個也是會讓GPU等待的 因為你要前一個函式算完 才能交給下一個函式算 這個地方 也是不容易平行的 但你會發現 這些被 連續呼叫的函式 都是FA 既然是FA造成的問題 那我們能不能就直接 把FA拿掉呢 所以我們直接把FA拿掉 我們就說不要FA了 這個RNN呢 我們就把reflection的部分拿掉 HT等於HT-1加上ST過FB
              
                  39:45
                  然後我們就把H1到HT通通寫出來 H1是H0加S1過FB H2是H1加S2過FB 以此類推 然後呢 我們再把H1帶進去 把H2帶進去 把H之間的dependency拿掉 那我們計算出來的結果就是 H1是X1過FB 那H2是X1過FB X2過FB H3是X1過FB X2過FB X3過FB 然後看起來就比剛才的式子簡單多了 所以今天如果要算HT的話 他就是T項 相加把X1到XT分別都過FB以後 加起來就是HT了 好,那像這樣子的式子有沒有辦法平行呢? 其實是有一些加速的方法 比如說你可以用一個叫Scan Algorithm的東西 把這一排數字用一個更有效的方法把它算出來 但是這個部分我們就今天就先不講
              
                  40:52
                  因為其實還有更有效、更簡化、更能平行的做法 怎麼做呢? 我們先簡化一下式子 我們先簡化一下我們的符號 那我們先假設HT呢 是一個D乘以D的矩陣 H可以是任何東西 我們這邊就假設 它是一個D乘以D的矩陣 那既然H是一個矩陣 那FB of X也得是一個矩陣 他們這樣才加得起來 那我們把FB of XT 用大D下標T來表示 好 那我們就知道說 H1就是D1 H2就是D 就是D1加D2 H3是D1D2加到D3 HT就是D1加D2加到DT 我們現在再做一下簡化 我們假設FC of HT 就是把HT這個矩陣 跟一個向量QT做相乘 然後QT呢 是輸入XT 因為這個QT呢 它是跟T有關的
              
                  41:54
                  那它怎麼來的呢 它是把XT乘上一個矩陣 WQ得到QT QT乘上 memory裡面的東西 HT就得到最終的輸出Y 所以Y1就是D1乘Q1 Y2就是D1乘Q2加D2乘Q2 Y3就是D1乘Q3 加到D3乘Q3 YT就是D1乘QT 加到DT乘QT 其實這一切都還有更有效的簡化方法 什麼樣的簡化方法呢 我們把DT寫成VT乘上KT V跟K分別是兩個向量 把V這個向量乘上K這個向量的transpose 就是展開變成一個矩陣 所以一個向量乘上另外一個向量transpose 你得到一個矩陣我們叫做DT 而VT跟KT是跟XT有關的 XT乘上WV得到VT XT乘上WK得到KT
              
                  42:59
                  然後DT就是VT乘上K的transpose 所以我們現在可以把所有的D通通都置換掉 所以Y1就是V1K1的transpose乘Q1 Y2就是V1K1的transpose乘Q2 加V2K2的transpose乘Q2 一直到YT就是V1K1的transpose乘QT 加到VTKT的transpose乘QT 講到這邊你有沒有發現了什麼 這個就是attention啊 我們已經把符號把QKV通通放到式子裡面了 如果你還沒有意會到為什麼它就是self-attention的話 那我們再繼續做運算 剛才我們是先把V1K1算出來再乘Q 那能不能換一下計算的順序 先算K跟Q呢 先計算K的transpose乘以Q呢 K的transpose乘以Q是什麼 K的transpose乘以Q 一個向量的transpose乘以另外一個向量 假設他們dimension是一樣的 所以可以直接做相乘 那你得到的結果就會是一個scalar 我把K1QT命名為αT1
              
                  44:06
                  K2 transpose QT命名為αT2 以此類推 我們先把K跟Q相乘剩下V 那你得到的就是V1乘一個Scalar Alpha V2乘另外一個Scalar Alpha VT乘另外一個Scalar Alpha 如果你不喜歡Scalar放在向量後面 那就把Scalar放到向量前面 這個不就是對一個叫做 Vvector V1到VT的東西 做weighted sum嗎 這其實就是self-attention 他跟原來你知道的self-attention 唯一的不同只是 少了softmax 那因為他少了softmax 所以他跟原來的self-attention還是有點不同 所以他自己有一個名字 叫做linear attention 好 所以講到這邊 我們知道了什麼 講到這邊 我們知道說 linear attention 就是RNN 拿掉reflection 我們把RNN拿掉FA拿掉reflection 就變成linear attention 而linear attention 就是self attention 沒有softmax
              
                  45:10
                  就是這麼神奇 那我們知道linear attention 就是self attention 沒有softmax以後有什麼好處呢 那linear attention 這樣子的RNN 他就是influence的時候 你用他來產生東西的時候 他像是一個RNN 在訓練的時候 你就展開 把他當transformer來train 他就transformer少了softmax啊 他完全可以套用 transformer的平行化的方法 來直接加速訓練的過程 所以假設剛才的 講attention的時候 到底怎麼加速的 你其實沒有聽得很清楚的話 也沒有關係 你只要記得 self attention是一個可以加速的東西 長得像self attention一樣 類似的這個level架構 都可以用類似的方法來加速 而linear attention 根本就是self attention 拿掉softmax 所以self attention 能怎麼展開 能怎麼加速 linear attention 就能怎麼展開 能怎麼加速 訓練的時候 像是一個self attention 但是influence的時候 他就像是一個RNN 就是這麼神奇
              
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
              
            