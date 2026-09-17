%%%
  VERSION:1
  LANGUAGE:ENGLISH
%%%
MODULE DemoModule
    ! Tool: straight torch, 150 mm along the flange Z axis
    PERS tooldata tTorch := [TRUE,[[0,0,150],[1,0,0,0]],[1.5,[0,0,50],[1,0,0,0],0,0,0]];
    TASK PERS wobjdata wTable := [FALSE,TRUE,"",[[400,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];
    CONST jointtarget jHome := [[0,-90,90,0,90,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
    CONST robtarget pApproach := [[150,-200,450],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
    CONST robtarget p10 := [[150,-200,250],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
    CONST robtarget p20 := [[300,-200,250],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
    CONST robtarget p30 := [[300,0,250],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
    CONST robtarget pCir := [[225,75,250],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
    CONST robtarget p40 := [[150,0,250],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]];
    CONST robtarget path{3} := [
        [[200,-75,300],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]],
        [[200,0,300],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]],
        [[200,75,300],[0,0,1,0],[0,0,0,0],[9E9,9E9,9E9,9E9,9E9,9E9]]];
    CONST speeddata vWeld := [150,500,5000,1000];
    CONST zonedata zSmall := [FALSE,5,8,8,0.8,8,0.8];

    PROC main()
        ConfL \Off;
        MoveAbsJ jHome\NoEOffs, v1000, z50, tool0;
        MoveJ pApproach, v1000, z50, tTorch \WObj:=wTable;
        MoveL p10, v500, fine, tTorch \WObj:=wTable;
        AccSet 50, 100;
        MoveL p20, vWeld, z20, tTorch \WObj:=wTable;
        MoveL p30, vWeld, zSmall, tTorch \WObj:=wTable;
        MoveC pCir, p40, vWeld, z10, tTorch \WObj:=wTable;
        MoveL Offs(p40, 0, 0, 100), v200 \V:=250, z20, tTorch \WObj:=wTable;
        AccSet 100, 100;
        WaitTime 0.5;
        FOR i FROM 1 TO 3 DO
            MoveL path{i}, v300, z10, tTorch \WObj:=wTable;
        ENDFOR
        MoveL RelTool(path{3}, 0, 0, -50 \Rz:=45), v100 \T:=2, fine, tTorch \WObj:=wTable;
        Retract;
        MoveAbsJ jHome\NoEOffs, v1000, fine, tool0;
    ENDPROC

    PROC Retract()
        MoveL Offs(CRobT(\Tool:=tTorch \WObj:=wTable), 0, 0, 100), v500, z50, tTorch \WObj:=wTable;
    ENDPROC
ENDMODULE
