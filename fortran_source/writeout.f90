module io

  contains

  subroutine writestuff()
    implicit none
    write(*,*)
    write(*,*) "     _            _   _"
    write(*,*) "    | |          | | (_)"
    write(*,*) "    | |_ ___  ___| |  _ _ __   __ _"
    write(*,*) "    | __/ _ \/ __| __| | '_ \ / _` |"
    write(*,*) "    | ||  __/\__ \ |_| | | | | (_| |"
    write(*,*) "     \__\___||___/\__|_|_| |_|\__, |"
    write(*,*) "                               __/ |"
    write(*,*) "                              |___/"

    call flush() ! this is important
  end subroutine writestuff

end module io
