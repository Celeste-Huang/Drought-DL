tas_daily_1901-1901
for tm=[0:1379]
    spei_ori=nc_varget('spei01.nc','spei',[tm 0 0],[1 360 720]); %1 month
    spei=spei_ori;
    for i=1:360
        spei(i,:)=spei_ori((361-i),:);
    end
    tas_ori=tas;
    for i=1:360
        tas(i,:)=tas_ori((361-i),:);
    end
    tas_ori=tas;
    tas(:,[1:360])=tas(:,[361:720]);
    tas(:,[361:720])=tas_ori(:,[1:360]);
    for i=1:720
        tas(:,i)=tas_ori(:,(721-i));
    end
    temp_ori=nc_varget(['data-temp/tas_daily_',num2str(1901+i),'-',num2str(1901+i),'.nc'])
    for tp=[0:]
    save(['SPEI2/spei-',num2str(ceil((tm+1)/12)+1900),'-',num2str(mod(tm,12)+1)],'spei','-ascii');
end